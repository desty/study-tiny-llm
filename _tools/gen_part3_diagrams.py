"""Part 3 SVG diagrams."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svg_prim import (
    svg_header, svg_footer, text_title, text_subtitle,
    node, group_around_nodes, arrow_line, arrow_path,
    arrow_legend, role_legend,
)

BASE = str(HERE.parent / 'docs' / 'assets' / 'diagrams')
os.makedirs(BASE, exist_ok=True)


def save(name, light_svg, dark_svg):
    with open(f'{BASE}/{name}.svg', 'w') as f:
        f.write(light_svg)
    with open(f'{BASE}/{name}-dark.svg', 'w') as f:
        f.write(dark_svg)
    os.system(f'rsvg-convert -w 1920 {BASE}/{name}.svg -o {BASE}/{name}.png 2>/dev/null')
    print(f'  ok {name}')


# =====================================================================
# Ch 7. Scaled dot-product attention 흐름
# =====================================================================

def attention_sdpa(theme):
    CW, CH = 1040, 380
    NW, NH = 130, 80
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'Scaled Dot-Product Attention — 5단계', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'Q · Kᵀ → ÷√d_k → mask → softmax → × V', theme))

    y = 130
    stages = [
        ('input', 'Q, K, V',     '(B,T,d_k)',     '입력 3 텐서'),
        ('model', 'Q · Kᵀ',      'matmul',        '(B,T,T) 점수'),
        ('gate',  '÷ √d_k',      'scale',         '분산 안정'),
        ('error', 'causal mask', '−∞ 위쪽',       '미래 차단'),
        ('llm',   'softmax',     'dim=-1',        '확률 분포'),
        ('output','× V',         'matmul',        '(B,T,d_k) 출력'),
    ]
    n = len(stages)
    gap = 18
    total = n * NW + (n - 1) * gap
    left = (CW - total) // 2
    xs = [left + i * (NW + gap) for i in range(n)]

    cy = y + NH // 2
    for i in range(n - 1):
        lines.extend(arrow_line(xs[i] + NW + 2, cy, xs[i + 1] - 2, cy, theme, kind='primary'))
    for x, (role, t, s, d) in zip(xs, stages):
        lines.extend(node(x, y, NW, NH + 20, role, theme, title=t, sub=s, detail=d))

    # Bottom note: F.scaled_dot_product_attention 한 줄
    note_y = 290
    lines.extend(text_subtitle(CW // 2, note_y, 'PyTorch ≥ 2.0:  out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)', theme, size=12))
    lines.extend(text_subtitle(CW // 2, note_y + 22, '(내부적으로 FlashAttention 자동 선택 — 같은 결과, 더 빠름)', theme, size=11))

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 8. 4개 현대 블록 비교
# =====================================================================

def modern_blocks(theme):
    CW, CH = 1040, 460
    NW, NH = 220, 100
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '2017 → 2024 — 표준이 된 4 블록', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '각각 코드 5~10줄 차이로 효율·메모리·일반화 한 축에서 이김', theme))

    # 4 cards in a 2x2 grid
    blocks = [
        # role,    title,        before,             after,                   why
        ('input',  '위치 인코딩',  'sinusoidal PE',     'RoPE',                  '길이 외삽 가능'),
        ('gate',   '정규화',       'LayerNorm',         'RMSNorm',               '7~10% 빠름, 같은 성능'),
        ('llm',    'FFN 활성',     'GeLU',              'SwiGLU',                '게이팅으로 표현력 ↑'),
        ('memory', 'Attention',    'MHA (Q=K=V head)',  'GQA (KV head ↓)',       '추론 KV cache 1/4'),
    ]
    gap_x = 30
    gap_y = 24
    cols = 2
    total_w = cols * NW + (cols - 1) * gap_x
    left = (CW - total_w) // 2
    top = 110

    for i, (role, title, before, after, why) in enumerate(blocks):
        col = i % cols
        row = i // cols
        x = left + col * (NW + gap_x)
        y = top + row * (NH + 50 + gap_y)
        # Card body
        lines.extend(node(x, y, NW, NH + 50, role, theme, title=title, sub=f'{before}  →  {after}', detail=why))

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('attention-sdpa', attention_sdpa('light'), attention_sdpa('dark'))
    save('modern-blocks',  modern_blocks('light'),  modern_blocks('dark'))
    print('Done.')
