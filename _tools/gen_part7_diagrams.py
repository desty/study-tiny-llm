"""Part 7 SVG diagrams."""
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
# Ch 24. LoRA 구조 — W + BA
# =====================================================================

def lora_structure(theme):
    CW, CH = 980, 460
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'LoRA — 작은 두 행렬의 곱으로 ΔW 근사', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'W (동결) + B·A (학습) — r 만 작으면 파라미터 1% 미만', theme))

    pal = P(theme)
    t = T(theme)

    # 왼쪽: 표준 SFT
    lx = 100
    ly = 130
    lines.extend(text_subtitle(lx + 130, ly - 12, 'Full SFT', theme, size=13))
    # W (큰 박스, 학습)
    lines.append(f'  <rect x="{lx}" y="{ly}" width="240" height="220" rx="10" fill="{pal["model"]["fill"]}" stroke="{pal["model"]["stroke"]}" stroke-width="2"/>')
    lines.append(f'  <text x="{lx + 120}" y="{ly + 110}" text-anchor="middle" font-size="32" font-weight="700" fill="{pal["model"]["text"]}">W</text>')
    lines.append(f'  <text x="{lx + 120}" y="{ly + 150}" text-anchor="middle" font-size="13" fill="{pal["model"]["sub"]}" font-family="JetBrains Mono, monospace">d × d</text>')
    lines.append(f'  <text x="{lx + 120}" y="{ly + 180}" text-anchor="middle" font-size="12" fill="{pal["error"]["sub"]}">전체 학습 (100%)</text>')

    # 오른쪽: LoRA
    rx = 540
    ry = 130
    lines.extend(text_subtitle(rx + 200, ry - 12, 'LoRA — W 동결 + BA 학습', theme, size=13))

    # W (동결)
    lines.append(f'  <rect x="{rx}" y="{ry}" width="160" height="220" rx="10" fill="{pal["model"]["fill"]}" stroke="{pal["model"]["stroke"]}" stroke-width="2" stroke-dasharray="6,3"/>')
    lines.append(f'  <text x="{rx + 80}" y="{ry + 110}" text-anchor="middle" font-size="28" font-weight="700" fill="{pal["model"]["text"]}">W</text>')
    lines.append(f'  <text x="{rx + 80}" y="{ry + 145}" text-anchor="middle" font-size="11" fill="{pal["model"]["sub"]}" font-family="JetBrains Mono, monospace">d × d</text>')
    lines.append(f'  <text x="{rx + 80}" y="{ry + 175}" text-anchor="middle" font-size="11" font-weight="700" fill="{pal["error"]["sub"]}">동결 (frozen)</text>')

    # +
    lines.append(f'  <text x="{rx + 175}" y="{ry + 115}" font-size="36" font-weight="700" fill="{t["title"]}">+</text>')

    # B (학습)
    bx = rx + 200
    lines.append(f'  <rect x="{bx}" y="{ry + 60}" width="40" height="100" rx="6" fill="{pal["token"]["fill"]}" stroke="{pal["token"]["stroke"]}" stroke-width="2"/>')
    lines.append(f'  <text x="{bx + 20}" y="{ry + 115}" text-anchor="middle" font-size="22" font-weight="700" fill="{pal["token"]["text"]}">B</text>')
    lines.append(f'  <text x="{bx + 20}" y="{ry + 175}" text-anchor="middle" font-size="10" fill="{pal["token"]["sub"]}" font-family="JetBrains Mono, monospace">d×r</text>')

    # ·
    lines.append(f'  <text x="{bx + 50}" y="{ry + 115}" font-size="22" fill="{t["title"]}">·</text>')

    # A (학습)
    ax = bx + 65
    lines.append(f'  <rect x="{ax}" y="{ry + 100}" width="100" height="20" rx="4" fill="{pal["token"]["fill"]}" stroke="{pal["token"]["stroke"]}" stroke-width="2"/>')
    lines.append(f'  <text x="{ax + 50}" y="{ry + 115}" text-anchor="middle" font-size="16" font-weight="700" fill="{pal["token"]["text"]}">A</text>')
    lines.append(f'  <text x="{ax + 50}" y="{ry + 175}" text-anchor="middle" font-size="10" fill="{pal["token"]["sub"]}" font-family="JetBrains Mono, monospace">r×d</text>')

    # 아래 캡션
    cap_y = ry + 230
    lines.append(f'  <text x="{rx + 200}" y="{cap_y}" text-anchor="middle" font-size="13" fill="{pal["token"]["sub"]}">학습 파라미터 = 2·d·r ≈ 1%</text>')

    # 아래: 메모리·디스크 비교
    bot_y = 390
    lines.append(f'  <text x="{lx + 120}" y="{bot_y}" text-anchor="middle" font-size="11" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">메모리 100% · 어댑터 = 모델 전체</text>')
    lines.append(f'  <text x="{rx + 200}" y="{bot_y}" text-anchor="middle" font-size="11" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">메모리 ~20% · 어댑터 ~20MB</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('lora-structure', lora_structure('light'), lora_structure('dark'))
    print('Done.')
