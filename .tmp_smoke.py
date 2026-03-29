import cv2
import time
from src.main import ParkModPipeline
from src.roi_manager import Zone
from src.email_notifier import EMAIL_CONFIG

EMAIL_CONFIG['enabled'] = False
pipe = ParkModPipeline(demo_mode=False)

# Force one active test zone covering full frame so timer/violation can be exercised.
pipe.roi.zones = [
    Zone(
        id='Z-TEST01',
        name='Test Full Frame',
        zone_type='rectangle',
        polygon=[[0, 0], [1919, 0], [1919, 1079], [0, 1079]],
        threshold=2.0,
        location='SmokeTest',
        department='QA',
        active=True,
        color=[0,255,100],
    )
]

cap = cv2.VideoCapture('data/input_videos/Different Types of Number Plates in Indian Vehicles #india #carshort #automobile (1) (1).mp4')
if not cap.isOpened():
    print('SMOKE_FAIL: unable to open test video')
    raise SystemExit(2)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

frame_idx = 0
max_frames = 220
max_tracks = 0

while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if pos_sec <= 0:
        pos_sec = frame_idx / fps
    _ = pipe.process_frame(frame, pos_sec)
    frame_idx += 1
    if pipe.engine.states:
        max_tracks = max(max_tracks, len(pipe.engine.states))

cap.release()

records = pipe.report.get_records()
print('frames_processed=', frame_idx)
print('states_seen=', len(pipe.engine.states))
print('max_parallel_tracks=', max_tracks)
print('violations=', len(records))
if records:
    print('sample_violation=', records[0])
print('alpr_summary=', pipe.alpr.summary())
