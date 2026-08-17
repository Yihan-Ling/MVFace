"""oct_head_pose_publisher.py — bridge MVFace head pose -> RAOCTg2 pupil tracker

Turns each `head_pose.estimate_head_pose(...)` result into the two DDS messages
that `PupilTrackingAction` (pupil_tracking.py) already subscribes to, so the
existing action moves the OCT with NO changes to pupil_tracking.py.

Publishes per frame:
  topic/face_features  (FaceDetectionList) : head orientation for the target face
  topic/pupil_features (PupilDetection)    : eye-target position (+ optional gaze)

Message shapes (RAOCTg2.types & igmr_robotics_toolkit.comms.types):
  FaceDetectionList(id, timestamp, frame, items[FaceDetection])
  FaceDetection(identity, rectangle:Rectangle2D, confidence, outline:Point2DArray, features:Optional[FaceFeaturePoses])
  FaceFeaturePoses(head:Pose, left_eye:Pose, right_eye:Pose)
  PupilDetection(id, timestamp, frame, position:Optional[Point3D], optical_axis:Optional[Vector3D])
  Pose : ndarray wrapper, Pose.from_numpy(4x4) / .to_numpy(); np.asanyarray works
  Point3D/Vector3D : (x, y, z) floats

Frame + units requirements:
  - FRAME must be a frame name registered in the RAOCTg2 frame graph relative to the robot base. 
    The action calls resolver.multi_resolve(frame, timestamps, relative_to=robot_base) per stamp.
  - Everything is METRES, board frame.
  - timestamp must be in the ROBOT's clock and should be the frame capture time,
    so pass the RealSense frame's host-clock capture time to publish().
"""
from __future__ import annotations

from time import time
from typing import Optional

import numpy as np
import torch

from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.topic import Topic
from cyclonedds.qos import Qos, Policy

from RAOCTg2.types import (PupilDetection, FaceDetectionList,
                           FaceDetection, FaceFeaturePoses)
from igmr_robotics_toolkit.comms.types import (Pose, Point3D, Point2D,
                                               Point2DArray, Rectangle2D)

from head_pose import estimate_head_pose, head_axes

# empty 2D outline reused for every message (the action ignores it)
_EMPTY_OUTLINE = Point2DArray.from_numpy(np.zeros((0, 2), dtype=np.float64))
_DUMMY_RECT = Rectangle2D(tl=Point2D(x=0.0, y=0.0), br=Point2D(x=0.0, y=0.0))


def _pose_from(R: np.ndarray, t: np.ndarray) -> Pose:
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = t
    return Pose.from_numpy(H)


class OCTHeadPosePublisher:
    def __init__(self,
                 ref_landmarks_path: str,
                 frame: str,
                 face_topic: str,
                 pupil_topic: str,
                 target_face: int = 0,
                 target_eye: str = "mid",        # 'mid' | 'left' | 'right'
                 rmsd_gate_m: float = 0.005,     # drop fits worse than 5 mm RMSD
                 publish_gaze: bool = False,     # keep False; use pupil_axis_from_face
                 allow_scale: bool = True,
                 domain_id: int = 0,
                 participant: Optional[DomainParticipant] = None):
        """
        Args:
            ref_landmarks_path: path to assets/mean_face_68.npy (68x3, metres).
            frame: board/rig frame name, as registered vs the robot base.
            face_topic / pupil_topic: EXACT topic name strings the action uses,
                i.e. the resolved values of params topic/face_features and
                topic/pupil_features. Must match on the wire or nothing connects.
            target_face: must equal the action's control/target_face (default 0).
            target_eye: which point the OCT aims at. 'mid' = interocular midpoint;
                for single-eye OCT set 'left' or 'right'.
            rmsd_gate_m: skip publishing frames whose rigid-fit RMSD exceeds this
                (metres); guards against garbage landmarks moving the arm.
            publish_gaze: if True, also publish optical_axis from the face normal.
                Prefer False + set robot/pupil_axis_from_face=True on the action,
                which derives the approach axis from the head orientation and
                avoids the gaze sign convention entirely (see note in publish()).
            domain_id / participant: must match the robot's DDS domain. Pass the
                robot's existing participant if you have one; else one is created
                on domain_id (must equal the robot's).
        """
        if target_eye not in ("mid", "left", "right"):
            raise ValueError(f"target_eye must be mid/left/right, got {target_eye!r}")

        self.ref = torch.from_numpy(np.load(ref_landmarks_path)).float()  # (68,3)
        self.frame = frame
        self.target_face = int(target_face)
        self.target_eye = target_eye
        self.rmsd_gate = float(rmsd_gate_m)
        self.publish_gaze = bool(publish_gaze)
        self.allow_scale = bool(allow_scale)
        self._seq = 0   # uint64 message id, increments per published frame

        dp = participant if participant is not None else DomainParticipant(domain_id)
        pub = Publisher(dp)
        qos = Qos(Policy.History.KeepLast(10))
        self._face_writer = DataWriter(
            pub, Topic(dp, face_topic, FaceDetectionList), qos)
        self._pupil_writer = DataWriter(
            pub, Topic(dp, pupil_topic, PupilDetection), qos)

    # -- message construction --
    def _build_face_list(self, R: np.ndarray, eyes_np: dict,
                         seq: int, stamp: float) -> FaceDetectionList:
        # The action uses only head's [:3,:3]. left/right filled meaningfully
        # (same orientation, eye-centre translations) since the type requires them.
        features = FaceFeaturePoses(
            head=_pose_from(R, eyes_np["mid"]),
            left_eye=_pose_from(R, eyes_np["left"]),
            right_eye=_pose_from(R, eyes_np["right"]),
        )
        face = FaceDetection(
            identity=self.target_face,
            rectangle=_DUMMY_RECT,
            confidence=1.0,
            outline=_EMPTY_OUTLINE,
            features=features,
        )
        return FaceDetectionList(id=seq, timestamp=float(stamp),
                                 frame=self.frame, items=[face])

    def _build_pupil(self, pos3: np.ndarray, axis3: Optional[np.ndarray],
                     seq: int, stamp: float) -> PupilDetection:
        position = Point3D(x=float(pos3[0]), y=float(pos3[1]), z=float(pos3[2]))
        optical_axis = None
        if axis3 is not None:
            optical_axis = Point3D(x=float(axis3[0]), y=float(axis3[1]),
                                   z=float(axis3[2]))
        return PupilDetection(id=seq, timestamp=float(stamp), frame=self.frame,
                              position=position, optical_axis=optical_axis)

    # -- main entry point --
    def publish(self, landmarks_3d: torch.Tensor,
                timestamp: Optional[float] = None) -> bool:
        """Estimate head pose from one frame's landmarks and publish both messages.

        Args:
            landmarks_3d: (68,3) or (1,68,3) predicted landmarks, board frame, metres.
            timestamp: frame capture time in the robot's clock. Defaults to time().
        Returns:
            True if published, False if the frame was gated out (bad fit).
        """
        if landmarks_3d.dim() == 3:
            if landmarks_3d.shape[0] != 1:
                raise ValueError("publish() expects a single frame, got a batch")
            landmarks_3d = landmarks_3d[0]

        stamp = time() if timestamp is None else float(timestamp)

        pose = estimate_head_pose(landmarks_3d, self.ref,
                                  allow_scale=self.allow_scale)
        if float(pose["rmsd"]) > self.rmsd_gate:
            return False

        R = pose["R"].detach().cpu().numpy().astype(np.float64)   # (3,3)
        eyes = pose["eyes"]
        eyes_np = {k: eyes[k].detach().cpu().numpy().astype(np.float64)
                   for k in ("mid", "left", "right")}

        seq = self._seq
        self._seq = (self._seq + 1) & ((1 << 64) - 1)   # wrap as uint64

        self._face_writer.write(self._build_face_list(R, eyes_np, seq, stamp))

        axis = None
        if self.publish_gaze:
            # head_axes = R @ +z = facial normal / approach direction in world.
            # The action sets eye_z = -optical_axis, so it wants optical_axis to
            # point OUTWARD (away from the OCT) -- hence the negation. This sign is
            # easy to get wrong; prefer robot/pupil_axis_from_face=True and leave
            # publish_gaze=False.
            fwd = head_axes(pose["R"]).detach().cpu().numpy().astype(np.float64)
            axis = -fwd

        self._pupil_writer.write(
            self._build_pupil(eyes_np[self.target_eye], axis, seq, stamp))
        return True

