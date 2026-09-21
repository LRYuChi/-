"""字形比對辨識：字模建立與準確率評估（實驗用，未併入 index.html）

2026-09-21 實測結果（三張已校準圖面，留一驗證）：
    字元正確率 93.4%   整串正確率 52.9%
    in-sample 字元 98.3%、整串 70.6%  ← 比對器沒問題，瓶頸在切字
    加上格式檢查與信心門檻 0.50 後：覆蓋 54.9%、精確率 89.3%

判定：不足以上線。精確率 89% 代表每 9 筆自動填入就有 1 筆錯，
使用者仍須逐筆核對，省不了時間又增加誤簽風險。

兩個獨立瓶頸：
  1. 切字——遮罩含公差堆疊的欄位（D_hole／H_total／D_base）會爆量切碎，
     R1 則會黏成一個。即使比對完美，整串上限也只有約 76%。
  2. 字模樣本太少——'9'、'R'、'+' 各只有 1 個樣本。
     即使切字修好，0.934^6 ≈ 66%，仍達不到 90%。

用法：python3 tools/glyph-eval.py
"""
import json, re, sys
from PIL import Image
import numpy as np

ASSETS = "/home/user/-/assets/"
GW, GH = 16, 24           # 正規化後的字形格子

# 每張圖每個欄位的真值（主尺寸含前綴；公差兩行）
TRUTH = {
"D4P001000001": {
 "D_outer":("Ø59.00",None,None), "D_mid":("Ø47.00",None,None), "D_pilot":("Ø20",None,None),
 "D_hole":("Ø9.73","+0.01","+0"), "R_nose":("R0.5",None,None),
 "H1":("25",None,None), "H2":("34.00",None,None), "H3":("24.00",None,None), "H4":("33.00",None,None),
 "H_total":("111.00","+0.10","-0.10"), "H5":("75",None,None),
 "D_neck":("Ø22.20",None,None), "D_base":("Ø50.00","-0.03","-0.05")},
"B4P001000001": {
 "D_outer":("Ø43.00",None,None), "D_mid":("Ø34.00",None,None), "D_pilot":("Ø18",None,None),
 "D_hole":("Ø7.76","+0.01","+0"), "R_nose":("R1",None,None),
 "H1":("20",None,None), "H2":("27.0",None,None), "H3":("20.00",None,None), "H4":("20.00",None,None),
 "H_total":("80.00","+0.10","-0.10"), "H5":("48",None,None),
 "D_neck":("Ø22.20",None,None), "D_base":("Ø40.00","-0.03","-0.05")},
"C4P001000002": {
 "D_outer":("Ø54.00",None,None), "D_mid":("Ø42.00",None,None), "D_pilot":("Ø20",None,None),
 "D_hole":("Ø9.73","+0.01","+0"), "R_nose":("R1",None,None),
 "H1":("20",None,None), "H2":("30.00",None,None), "H3":("21.00",None,None), "H4":("43.00",None,None),
 "H_total":("110.00","+0.10","-0.10"), "H5":("75",None,None),
 "D_neck":("Ø22.20",None,None), "D_base":("Ø45.00","-0.03","-0.05")},
}

def load_fields():
    """從 index.html 讀出每個欄位的遮罩與方向"""
    src = open("/home/user/-/index.html", encoding="utf8").read()
    blocks = re.findall(r'id:"([A-Z0-9]+)",\s*name:"[^"]+".*?fields:\[(.*?)\n    \]', src, re.S)
    out = {}
    for name, fsrc in blocks:
        fields = {}
        for e in re.findall(r'field\(\{(.*?)\}\)', fsrc, re.S):
            fid = re.search(r'id:"([^"]+)"', e).group(1)
            m = [int(v) for v in re.search(r'mask:px\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)', e).groups()]
            ro = re.search(r'orientation:(-?\d+)', e)
            fields[fid] = dict(mask=m, rot=int(ro.group(1)) if ro else 0)
        if fields: out[name] = fields
    return out

def components(g, min_area=4):
    """8 鄰接連通元件"""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    for sy in range(h):
        for sx in range(w):
            if not g[sy, sx] or seen[sy, sx]: continue
            stack = [(sy, sx)]; seen[sy, sx] = True
            minx = maxx = sx; miny = maxy = sy; area = 0
            px = []
            while stack:
                y, x = stack.pop(); area += 1; px.append((y, x))
                minx = min(minx, x); maxx = max(maxx, x)
                miny = min(miny, y); maxy = max(maxy, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and g[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
            if area >= min_area:
                comps.append(dict(x=minx, y=miny, w=maxx-minx+1, h=maxy-miny+1, area=area, px=px))
    return comps

def split_groups(comps):
    """分出主尺寸與公差：先用字高分大小，再把標點依鄰近歸隊"""
    if not comps: return [], []
    size = lambda c: max(c["w"], c["h"])
    hi = max(size(c) for c in comps)
    digits = [c for c in comps if size(c) >= 0.40*hi]
    tiny   = [c for c in comps if size(c) <  0.40*hi]
    main = [c for c in digits if size(c) >= 0.70*hi]
    tol  = [c for c in digits if size(c) <  0.70*hi]
    def bbox(arr):
        return (min(c["x"] for c in arr), min(c["y"] for c in arr),
                max(c["x"]+c["w"] for c in arr), max(c["y"]+c["h"] for c in arr))
    for t in tiny:                       # 標點歸給比較近的那一群
        cx, cy = t["x"]+t["w"]/2, t["y"]+t["h"]/2
        def dist(arr):
            if not arr: return 1e9
            x0,y0,x1,y1 = bbox(arr)
            dx = max(x0-cx, 0, cx-x1); dy = max(y0-cy, 0, cy-y1)
            return (dx*dx+dy*dy)**.5
        (main if dist(main) <= dist(tol) else tol).append(t)
    return main, tol

def order(comps, vertical=False):
    return sorted(comps, key=lambda c: c["x"] + c["y"]*0.001)

def drop_lines(comps, W, H):
    """丟掉貫穿整個框、細長的線段（尺寸線、輪廓線）"""
    out = []
    for c in comps:
        long_side, short_side = max(c["w"], c["h"]), max(1, min(c["w"], c["h"]))
        aspect = long_side/short_side
        spans = (c["w"] >= 0.8*W) or (c["h"] >= 0.8*H)
        if aspect >= 8 and spans: continue
        out.append(c)
    return out

def split_rows(comps):
    """公差兩行：依 y 中心分群，間距門檻用字高推算"""
    if not comps: return []
    cs = sorted(comps, key=lambda c: c["y"]+c["h"]/2)
    med_h = sorted(c["h"] for c in cs)[len(cs)//2]
    rows, cur = [], [cs[0]]
    for c in cs[1:]:
        prev_cy = sum(p["y"]+p["h"]/2 for p in cur)/len(cur)
        if (c["y"]+c["h"]/2) - prev_cy > med_h*0.8: rows.append(cur); cur = [c]
        else: cur.append(c)
    rows.append(cur)
    return [order(r) for r in rows]

def _union(p, c):
    nx, ny = min(p["x"],c["x"]), min(p["y"],c["y"])
    nx2, ny2 = max(p["x"]+p["w"],c["x"]+c["w"]), max(p["y"]+p["h"],c["y"]+c["h"])
    return dict(x=nx, y=ny, w=nx2-nx, h=ny2-ny, area=p["area"]+c["area"], px=p["px"]+c["px"])

def fix_fragments(comps):
    """把同一個字被掃描切斷的碎片合回去：x 重疊要夠多，且合併後高度不能超過正常字高"""
    if len(comps) < 2: return comps
    cs = order(comps)
    med_h = sorted(c["h"] for c in cs)[len(cs)//2]
    out = [cs[0]]
    for c in cs[1:]:
        p = out[-1]
        ov = min(p["x"]+p["w"], c["x"]+c["w"]) - max(p["x"], c["x"])
        big = max(p["w"], c["w"])
        u = _union(p, c)
        if ov >= 0.6*big and u["h"] <= med_h*1.35:
            out[-1] = u
        else:
            out.append(c)
    return out

def split_wide(comps, g):
    """把黏在一起的兩個字切開：寬度明顯超過中位數者，在中段墨跡最少的位置切"""
    if not comps: return comps
    hs = sorted(c["h"] for c in comps)
    med_h = hs[len(hs)//2]
    digitish = [c["w"] for c in comps if c["h"] >= med_h*0.7]
    med_w = sorted(digitish)[len(digitish)//2] if digitish else max(6, int(med_h*0.62))
    med_w = max(med_w, int(med_h*0.45))      # 下限：避免被標點拉低
    out = []
    for c in comps:
        if c["w"] <= med_w*1.55 or c["w"] < 8:
            out.append(c); continue
        n = min(4, max(2, int(round(c["w"]/med_w))))
        sub = g[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]]
        colsum = sub.sum(axis=0)
        cuts = []
        for k in range(1, n):
            target = int(c["w"]*k/n)
            lo, hi = max(1, target-max(2,int(med_w*0.3))), min(c["w"]-1, target+max(2,int(med_w*0.3)))
            if hi <= lo: continue
            cuts.append(lo + int(np.argmin(colsum[lo:hi])))
        bounds = [0] + sorted(set(cuts)) + [c["w"]]
        for a, b in zip(bounds, bounds[1:]):
            if b-a < 3: continue
            piece = sub[:, a:b]
            ys = np.where(piece.any(axis=1))[0]
            xs = np.where(piece.any(axis=0))[0]
            if not len(ys) or not len(xs): continue
            out.append(dict(x=c["x"]+a+int(xs[0]), y=c["y"]+int(ys[0]),
                            w=int(xs[-1]-xs[0]+1), h=int(ys[-1]-ys[0]+1),
                            area=int(piece.sum()), px=[]))
    return order(out)

def norm_glyph(g, c):
    """把一個字形正規化成 GW×GH 的位元圖"""
    sub = g[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]]
    im = Image.fromarray((sub*255).astype(np.uint8))
    # 等比縮放後置中
    s = min(GW/c["w"], GH/c["h"])
    nw, nh = max(1,int(round(c["w"]*s))), max(1,int(round(c["h"]*s)))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (GW, GH), 0)
    canvas.paste(im, ((GW-nw)//2, (GH-nh)//2))
    return (np.asarray(canvas) > 110).astype(np.uint8)

def region_glyphs(img, mask, rot):
    """回傳 (主尺寸字形清單, 公差各行字形清單)，每個字形是 (bitmap, comp)"""
    x, y, w, h = mask
    crop = img[y:y+h, x:x+w]
    pil = Image.fromarray((crop*255).astype(np.uint8))
    if rot == -90: pil = pil.rotate(-90, expand=True)
    elif rot == 90: pil = pil.rotate(90, expand=True)
    S = 3                                    # 放大後筆畫較粗，斷字與黏字都少很多
    pil = pil.resize((pil.width*S, pil.height*S), Image.LANCZOS)
    g = (np.asarray(pil) > 110).astype(np.uint8)
    H, W = g.shape
    comps = drop_lines(components(g), W, H)
    if not comps: return [], []
    main, tol = split_groups(comps)
    main = split_wide(fix_fragments(main), g)
    tol  = fix_fragments(tol)
    return ([(norm_glyph(g, c), c) for c in order(main)],
            [[(norm_glyph(g, c), c) for c in row] for row in split_rows(tol)])

# ---------------- 建庫 ----------------
def build(only=None, exclude=None):
    fields = load_fields()
    bank = {}      # char -> list of bitmaps
    stats = dict(ok=0, mismatch=[], total=0)
    for name, fmap in fields.items():
        if only and name != only: continue
        if exclude and name == exclude: continue
        img = (np.asarray(Image.open(ASSETS+name+".jpg").convert("L")) < 120).astype(np.uint8)
        for fid, cfg in fmap.items():
            truth = TRUTH[name].get(fid)
            if not truth: continue
            val, tu, tl = truth
            mains, tolrows = region_glyphs(img, cfg["mask"], cfg["rot"])
            stats["total"] += 1
            expect = [(val, mains)]
            tolstrs = [s for s in (tu, tl) if s]
            if len(tolrows) == len(tolstrs):
                expect += list(zip(tolstrs, tolrows))
            elif tolstrs:
                stats["mismatch"].append(f"{name}.{fid} 公差行數 {len(tolrows)}≠{len(tolstrs)}")
            good = True
            for s, glyphs in expect:
                if len(glyphs) != len(s):
                    stats["mismatch"].append(f"{name}.{fid} '{s}' 切出 {len(glyphs)} 個字")
                    good = False; break
            if not good: continue
            stats["ok"] += 1
            for s, glyphs in expect:
                for ch, (bm, _) in zip(s, glyphs):
                    bank.setdefault(ch, []).append(bm)
    return bank, stats

def blur(a):
    p = np.pad(a.astype(np.float32), 1)
    return (p[1:-1,1:-1]*4 + p[:-2,1:-1] + p[2:,1:-1] + p[1:-1,:-2] + p[1:-1,2:])/8.0

def match(bm, bank_blur):
    """回傳 (最佳字元, 信心)。信心 = (次佳距離-最佳距離)/最佳距離"""
    b = blur(bm)
    best = []
    for ch, arrs in bank_blur.items():
        d = min(float(((b-a)**2).sum()) for a in arrs)
        best.append((d, ch))
    best.sort()
    d1, c1 = best[0]
    d2 = best[1][0] if len(best) > 1 else d1*2
    conf = (d2-d1)/max(d1, 1e-6)
    return c1, conf, d1

def prep(bank):
    return {ch: [blur(b) for b in arr] for ch, arr in bank.items()}

# ---------------- 評估：留一張圖當測試 ----------------
if __name__ == "__main__":
    allbank, allstats = build()
    print("=== 切字結果（全部三張） ===")
    print(f"欄位總數 {allstats['total']}，切字與真值吻合 {allstats['ok']}")
    for m in allstats["mismatch"]: print("   ✗", m)
    print("字模庫：", {k: len(v) for k, v in sorted(allbank.items())})

    print("\n=== 留一驗證 ===")
    fields = load_fields()
    tot_ch = ok_ch = tot_fd = ok_fd = 0
    lowconf_saves = 0
    for held in TRUTH:
        bank, _ = build(exclude=held)
        bb = prep(bank)
        img = (np.asarray(Image.open(ASSETS+held+".jpg").convert("L")) < 120).astype(np.uint8)
        for fid, cfg in fields[held].items():
            truth = TRUTH[held].get(fid)
            if not truth: continue
            val, tu, tl = truth
            mains, tolrows = region_glyphs(img, cfg["mask"], cfg["rot"])
            strs = [(val, mains)]
            tolstrs = [s for s in (tu, tl) if s]
            if len(tolrows) == len(tolstrs): strs += list(zip(tolstrs, tolrows))
            for s, glyphs in strs:
                tot_fd += 1
                if len(glyphs) != len(s):
                    print(f"   ✗ {held}.{fid} '{s}' 切字數不符({len(glyphs)})")
                    continue
                got, confs = "", []
                for ch, (bm, _) in zip(s, glyphs):
                    c, conf, _ = match(bm, bb)
                    got += c; confs.append(conf); tot_ch += 1
                    if c == ch: ok_ch += 1
                if got == s: ok_fd += 1
                else:
                    mc = min(confs)
                    print(f"   ✗ {held}.{fid} 真值 '{s}' → 辨識 '{got}'  最低信心 {mc:.2f}")
                    if mc < 0.25: lowconf_saves += 1
    print(f"\n字元正確率 {ok_ch}/{tot_ch} = {ok_ch/max(tot_ch,1)*100:.1f}%")
    print(f"整串正確率 {ok_fd}/{tot_fd} = {ok_fd/max(tot_fd,1)*100:.1f}%")
