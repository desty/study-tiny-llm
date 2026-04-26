"""Part 4 SVG diagrams."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svg_prim import (
    svg_header, svg_footer, text_title, text_subtitle, node,
    arrow_line, P, T,
)

BASE = str(HERE.parent / 'docs' / 'assets' / 'diagrams')
os.makedirs(BASE, exist_ok=True)

def save(name, light, dark):
    open(f'{BASE}/{name}.svg', 'w').write(light)
    open(f'{BASE}/{name}-dark.svg', 'w').write(dark)
    os.system(f'rsvg-convert -w 1920 {BASE}/{name}.svg -o {BASE}/{name}.png 2>/dev/null')
    print(f'  ok {name}')


# =====================================================================
# Ch 14. 손실 곡선 5가지 패턴
# =====================================================================

def loss_patterns(theme):
    CW, CH = 1100, 480
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '손실 곡선 5가지 패턴', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '학습 진단의 가장 강한 시그널 — 곡선 모양으로 즉시 분류', theme))

    # 5 panels in a row
    panels = [
        ('output', '정상',     'cosine 따라 ↓',     'normal'),
        ('error',  '발산',     'NaN / 폭발',         'diverge'),
        ('memory', '정체',     'ln(vocab) 부근',    'plateau'),
        ('gate',   '스파이크', '한 번씩 ↑',          'spike'),
        ('llm',    '과적합',   'train ↓ val ↑',      'overfit'),
    ]
    n = 5
    PW, PH = 195, 280
    gap = 12
    total = n * PW + (n - 1) * gap
    left = (CW - total) // 2
    top = 95
    t = T(theme)
    pal = P(theme)

    for i, (role, title, sub, kind) in enumerate(panels):
        x = left + i * (PW + gap)
        # 카드 배경
        lines.append(f'  <rect x="{x}" y="{top}" width="{PW}" height="{PH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
        # 제목
        lines.append(f'  <text x="{x + PW//2}" y="{top + 24}" text-anchor="middle" font-size="14" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + PW//2}" y="{top + 42}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{sub}</text>')

        # plot 영역
        plot_top = top + 60
        plot_bot = top + PH - 30
        plot_left = x + 16
        plot_right = x + PW - 16
        plot_w = plot_right - plot_left
        plot_h = plot_bot - plot_top
        lines.append(f'  <rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="{t["bg"]}" opacity="0.5"/>')

        # axes
        lines.append(f'  <line x1="{plot_left}" y1="{plot_bot}" x2="{plot_right}" y2="{plot_bot}" stroke="{t["legend_text"]}" stroke-width="1"/>')
        lines.append(f'  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bot}" stroke="{t["legend_text"]}" stroke-width="1"/>')

        # 곡선
        path = ""
        if kind == 'normal':
            # 빠른 ↓ 후 평탄
            pts = []
            for j in range(20):
                px = plot_left + j * plot_w / 19
                # 지수 decay: 1.0 → 0.2
                py = plot_bot - plot_h * (0.2 + 0.6 * 2.71 ** (-j * 0.3))
                pts.append(f"{px:.0f},{py:.0f}")
            color = pal['output']['stroke']
            path = "M " + " L ".join(pts)
        elif kind == 'diverge':
            pts = []
            for j in range(15):
                px = plot_left + j * plot_w / 19
                py = plot_bot - plot_h * (0.4 + 0.05 * j)
                pts.append(f"{px:.0f},{py:.0f}")
            for j in range(15, 20):
                px = plot_left + j * plot_w / 19
                py = plot_top + 5    # 폭발 위쪽으로
                pts.append(f"{px:.0f},{py:.0f}")
            color = pal['error']['stroke']
            path = "M " + " L ".join(pts)
        elif kind == 'plateau':
            pts = [f"{plot_left + j * plot_w / 19:.0f},{plot_top + plot_h * 0.15:.0f}" for j in range(20)]
            color = pal['memory']['stroke']
            path = "M " + " L ".join(pts)
        elif kind == 'spike':
            import random
            random.seed(2)
            pts = []
            for j in range(20):
                px = plot_left + j * plot_w / 19
                base_y = plot_bot - plot_h * (0.2 + 0.6 * 2.71 ** (-j * 0.25))
                if j in [8, 14]:
                    base_y -= plot_h * 0.4    # 스파이크
                pts.append(f"{px:.0f},{base_y:.0f}")
            color = pal['gate']['stroke']
            path = "M " + " L ".join(pts)
        elif kind == 'overfit':
            # train (↓) 와 val (V 형)
            train_pts = []
            val_pts = []
            for j in range(20):
                px = plot_left + j * plot_w / 19
                ty = plot_bot - plot_h * (0.15 + 0.7 * 2.71 ** (-j * 0.3))
                # val 은 절반 후 ↑
                if j < 10:
                    vy = plot_bot - plot_h * (0.3 + 0.5 * 2.71 ** (-j * 0.3))
                else:
                    vy = plot_bot - plot_h * (0.45 - 0.02 * (j - 10))
                train_pts.append(f"{px:.0f},{ty:.0f}")
                val_pts.append(f"{px:.0f},{vy:.0f}")
            train_path = "M " + " L ".join(train_pts)
            val_path = "M " + " L ".join(val_pts)
            lines.append(f'  <path d="{train_path}" stroke="{pal["output"]["stroke"]}" stroke-width="2" fill="none"/>')
            lines.append(f'  <path d="{val_path}" stroke="{pal["error"]["stroke"]}" stroke-width="2" fill="none" stroke-dasharray="4,2"/>')
            lines.append(f'  <text x="{plot_right - 4}" y="{plot_top + 14}" text-anchor="end" font-size="9" fill="{pal["output"]["sub"]}">train</text>')
            lines.append(f'  <text x="{plot_right - 4}" y="{plot_top + 28}" text-anchor="end" font-size="9" fill="{pal["error"]["sub"]}">val</text>')
            path = ""

        if path:
            lines.append(f'  <path d="{path}" stroke="{color}" stroke-width="2" fill="none"/>')

        # x-axis label
        lines.append(f'  <text x="{x + PW//2}" y="{top + PH - 8}" text-anchor="middle" font-size="9" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">step →</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('loss-patterns', loss_patterns('light'), loss_patterns('dark'))
    print('Done.')
