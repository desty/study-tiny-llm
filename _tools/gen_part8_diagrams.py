"""Part 8 SVG diagrams."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svg_prim import (
    svg_header, svg_footer, text_title, text_subtitle, node,
    arrow_line, arrow_path, P, T,
)

BASE = str(HERE.parent / 'docs' / 'assets' / 'diagrams')
os.makedirs(BASE, exist_ok=True)

def save(name, light, dark):
    open(f'{BASE}/{name}.svg', 'w').write(light)
    open(f'{BASE}/{name}-dark.svg', 'w').write(dark)
    os.system(f'rsvg-convert -w 1920 {BASE}/{name}.svg -o {BASE}/{name}.png 2>/dev/null')
    print(f'  ok {name}')


# =====================================================================
# Ch 32. 운영 사이클 (분기)
# =====================================================================

def production_cycle(theme):
    CW, CH = 900, 600
    NW, NH = 180, 70
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '분기 운영 사이클 — 배포에서 다음 학습까지', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '3개월 한 사이클 · 자동화된 신호로 진행', theme))

    pal = P(theme); t = T(theme)
    cx = CW // 2
    cy = CH // 2 + 30
    radius = 200

    # 6 단계 원형 배치
    import math
    steps = [
        ('output', '배포',         'A/B → ramp'),
        ('gate',   '모니터링',     '4 시그널'),
        ('error',  '신호 발견',    '거절↑·드리프트'),
        ('input',  '데이터 수집',  '거절·재요청 케이스'),
        ('token',  '학습',         '재학습·LoRA'),
        ('llm',    '평가',         '회귀·OOD·A/B'),
    ]
    n = 6
    angle_offset = -math.pi / 2  # 12시 방향

    positions = []
    for i in range(n):
        angle = angle_offset + 2 * math.pi * i / n
        x = cx + radius * math.cos(angle) - NW // 2
        y = cy + radius * math.sin(angle) - NH // 2
        positions.append((x, y))

    # 원형 화살표
    for i in range(n):
        next_i = (i + 1) % n
        x1, y1 = positions[i]
        x2, y2 = positions[next_i]
        # 노드 중심
        c1 = (x1 + NW//2, y1 + NH//2)
        c2 = (x2 + NW//2, y2 + NH//2)
        # 직선 화살표 (간단)
        # 단축: 노드 가장자리 근처까지
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        d = (dx**2 + dy**2) ** 0.5
        margin = 50
        sx = c1[0] + dx * margin / d
        sy = c1[1] + dy * margin / d
        ex = c2[0] - dx * margin / d
        ey = c2[1] - dy * margin / d
        lines.extend(arrow_line(sx, sy, ex, ey, theme, kind='primary'))

    # 노드
    for (x, y), (role, title, sub) in zip(positions, steps):
        lines.extend(node(x, y, NW, NH + 14, role, theme, title=title, sub=sub))

    # 중앙 메시지
    lines.append(f'  <text x="{cx}" y="{cy - 6}" text-anchor="middle" font-size="14" font-weight="700" fill="{t["title"]}">분기 사이클</text>')
    lines.append(f'  <text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{t["subtitle"]}">3 개월 / 1 회</text>')

    # 롤백 화살표 (배포 ↔ 어댑터 swap)
    lines.append(f'  <text x="{cx}" y="{CH - 30}" text-anchor="middle" font-size="11" fill="{pal["error"]["sub"]}">↺ 사고 시 롤백 30초 (LoRA 어댑터 swap)</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('production-cycle', production_cycle('light'), production_cycle('dark'))
    print('Done.')
