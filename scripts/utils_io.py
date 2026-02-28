import glob, os, json
from . import settings

def latest_video(vdir):
    mp4 = glob.glob(os.path.join(vdir, "*.mp4"))
    if not mp4: raise FileNotFoundError("videos に mp4 がありません")
    return max(mp4, key=os.path.getmtime)

def load_m_per_px():
    try:
        with open(settings.CALIB_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d.get("m_per_px", settings.M_PER_PX_FALLBACK))
    except:
        return settings.M_PER_PX_FALLBACK







