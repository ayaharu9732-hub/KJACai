import cv2
import numpy as np

def draw_guides(frame):
    h, w = frame.shape[:2]
    cv2.line(frame, (w//2, 0), (w//2, h), (0,255,0), 1)
    cv2.line(frame, (0, h//2), (w, h//2), (0,255,0), 1)

def put_label(frame, text, org=(20,30)):
    cv2.putText(frame, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

def flow_metrics(prev, curr, dt, prev_pts=None):
    # Shi-Tomasi でコーナー点検出
    if prev_pts is None or len(prev_pts) < 50:
        prev_pts = cv2.goodFeaturesToTrack(prev,
                                           maxCorners=400,
                                           qualityLevel=0.01,
                                           minDistance=7)
    if prev_pts is None:
        return 0.0, None, 0.0, 0.0

    next_pts, st, err = cv2.calcOpticalFlowPyrLK(
        prev, curr, prev_pts, None,
        winSize=(31,31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,0.01))

    if next_pts is None:
        return 0.0, None, 0.0, 0.0

    good_new = next_pts[st==1]; good_old = prev_pts[st==1]
    if len(good_new) == 0:
        return 0.0, None, 0.0, 0.0

    flow = good_new - good_old
    dx, dy = flow[:,0], flow[:,1]
    dx_mean, dy_mean = np.mean(dx), np.mean(dy)
    v_px = np.sqrt(dx_mean**2 + dy_mean**2) / max(dt,1e-6)
    return float(v_px), good_new.reshape(-1,1,2), np.std(dx), np.std(dy)


def get_trunk_tilt_deg(frame, pose, RIGHT, LEFT):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if not results.pose_landmarks: return None
    lm = results.pose_landmarks.landmark
    r_sh, r_hp = lm[RIGHT["shoulder"]], lm[RIGHT["hip"]]
    l_sh, l_hp = lm[LEFT["shoulder"]], lm[LEFT["hip"]]
    sh = np.array([(r_sh.x+l_sh.x)/2, (r_sh.y+l_sh.y)/2])
    hp = np.array([(r_hp.x+l_hp.x)/2, (r_hp.y+l_hp.y)/2])
    v = hp - sh
    deg = np.degrees(np.arctan2(v[1], v[0]))
    return float(deg)


def classify_phase(v_px, v_peak_px, dv_px, tilt):
    if v_px < 20: return "静止/準備"
    if v_px < 0.7*v_peak_px: return "加速中"
    return "最高速～維持"
