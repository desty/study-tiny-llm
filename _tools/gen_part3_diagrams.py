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


# =====================================================================
# Ch 11. 학습 메모리 분해 스택 (params + grads + Adam + activation)
# =====================================================================

def memory_stack(theme):
    from svg_prim import P, T
    CW, CH = 1000, 460
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '학습 메모리 분해 — 14N + activation', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'bf16 mixed precision 기준 · 본 책 10M / 125M / 1B 비교', theme))

    pal = P(theme); t = T(theme)

    # 3 스택 (10M, 125M, 1B)
    cases = [
        ('10M', [('params (bf16)', 0.02, 'token'), ('grads', 0.02, 'gate'),
                 ('Adam m', 0.04, 'memory'), ('Adam v', 0.04, 'memory'),
                 ('activation', 0.84, 'model')]),
        ('125M', [('params', 0.25, 'token'), ('grads', 0.25, 'gate'),
                  ('Adam m', 0.5, 'memory'), ('Adam v', 0.5, 'memory'),
                  ('activation', 1.5, 'model')]),
        ('1B', [('params', 2.0, 'token'), ('grads', 2.0, 'gate'),
                ('Adam m', 4.0, 'memory'), ('Adam v', 4.0, 'memory'),
                ('activation', 8.0, 'model')]),
    ]

    # bar 영역
    bar_top = 110
    bar_bot = 360
    bar_h = bar_bot - bar_top
    bar_w = 140
    gap = 100
    n = len(cases)
    total = n * bar_w + (n - 1) * gap
    left = (CW - total) // 2

    # 스케일 — 모든 케이스 비례 비교 위해 max 기준
    max_total = max(sum(s[1] for s in segs) for _, segs in cases)

    for i, (name, segs) in enumerate(cases):
        x = left + i * (bar_w + gap)
        cur_total = sum(s[1] for s in segs)
        # 본 케이스의 stack 높이 (max 비례)
        case_h = bar_h * (cur_total / max_total)
        y_start = bar_bot - case_h

        # 모델 라벨
        lines.append(f'  <text x="{x + bar_w//2}" y="{bar_top - 10}" text-anchor="middle" font-size="14" font-weight="700" fill="{t["title"]}">{name}</text>')
        lines.append(f'  <text x="{x + bar_w//2}" y="{bar_bot + 22}" text-anchor="middle" font-size="11" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">{cur_total:.2f} GB</text>')

        # 세그먼트 쌓기
        cy = y_start
        for label, gb, role in segs:
            seg_h = bar_h * (gb / max_total)
            lines.append(f'  <rect x="{x}" y="{cy}" width="{bar_w}" height="{seg_h}" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.2"/>')
            # 라벨 (세그먼트 안에)
            if seg_h > 20:
                lines.append(f'  <text x="{x + bar_w//2}" y="{cy + seg_h/2 + 4}" text-anchor="middle" font-size="10" fill="{pal[role]["text"]}">{label}</text>')
                lines.append(f'  <text x="{x + bar_w//2}" y="{cy + seg_h/2 + 18}" text-anchor="middle" font-size="9" fill="{pal[role]["sub"]}" font-family="JetBrains Mono, monospace">{gb:.2f}GB</text>')
            cy += seg_h

    # 범례
    lines.extend(text_subtitle(CW // 2, 410, '* activation = batch · seq · hidden · layer · 14 (대략) · 2 byte', theme, size=10))
    lines.extend(text_subtitle(CW // 2, 432, '** B=32, T=512 가정 · gradient checkpointing 안 함', theme, size=10))

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 11. 메모리 비율 (100% 정규화)
# =====================================================================

def memory_breakdown_pct(theme):
    from svg_prim import P, T
    CW, CH = 1000, 460
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '학습 메모리 — 항별 비율 (모델 크기별)', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '같은 height = 100% 로 정규화 · 항 비중을 모델 간 비교', theme))

    pal = P(theme); t = T(theme)

    # 비율 (10M, 125M, 1B 순)
    cases = [
        ('10M', [('params', 0.02, 'token'), ('grads', 0.02, 'gate'),
                 ('Adam', 0.08, 'memory'), ('activation', 0.84, 'model')]),
        ('125M', [('params', 0.08, 'token'), ('grads', 0.08, 'gate'),
                  ('Adam', 0.34, 'memory'), ('activation', 0.50, 'model')]),
        ('1B', [('params', 0.10, 'token'), ('grads', 0.10, 'gate'),
                ('Adam', 0.40, 'memory'), ('activation', 0.40, 'model')]),
    ]

    bar_top = 110
    bar_bot = 380
    bar_h = bar_bot - bar_top
    bar_w = 180
    gap = 80
    n = len(cases)
    total = n * bar_w + (n - 1) * gap
    left = (CW - total) // 2

    for i, (name, segs) in enumerate(cases):
        x = left + i * (bar_w + gap)
        lines.append(f'  <text x="{x + bar_w//2}" y="{bar_top - 12}" text-anchor="middle" font-size="14" font-weight="700" fill="{t["title"]}">{name}</text>')

        cy = bar_top
        for label, pct, role in segs:
            seg_h = bar_h * pct
            lines.append(f'  <rect x="{x}" y="{cy}" width="{bar_w}" height="{seg_h}" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.2"/>')
            if seg_h > 18:
                lines.append(f'  <text x="{x + bar_w//2}" y="{cy + seg_h/2 + 4}" text-anchor="middle" font-size="11" fill="{pal[role]["text"]}">{label}</text>')
                lines.append(f'  <text x="{x + bar_w//2}" y="{cy + seg_h/2 + 18}" text-anchor="middle" font-size="9" fill="{pal[role]["sub"]}" font-family="JetBrains Mono, monospace">{int(pct*100)}%</text>')
            cy += seg_h

    lines.extend(text_subtitle(CW // 2, 410, '* 10M = activation 이 84% · 1B = activation 비중 ↓, Adam state ↑', theme, size=11))
    lines.extend(text_subtitle(CW // 2, 432, '** gradient checkpointing 사용 시 activation 비율 1/√L 로 ↓', theme, size=11))

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 10. nanoGPT 트랜스포머 블록
# =====================================================================

def nanogpt_block(theme):
    from svg_prim import P, T
    CW, CH = 900, 540
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'nanoGPT 블록 — Pre-norm + Residual 두 번', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'norm → sublayer → + 입력 (residual) — 두 번 반복', theme))

    pal = P(theme); t = T(theme)
    cx = CW // 2

    # 입력 x
    lines.extend(node(cx - 60, 100, 120, 50, 'input', theme, title='x (B,T,D)'))

    # Block 1: Norm → Attn → +
    by = 180
    lines.extend(node(cx - 200, by, 120, 50, 'gate', theme, title='RMSNorm'))
    lines.extend(node(cx - 60, by, 120, 50, 'model', theme, title='SelfAttn', sub='+ RoPE'))

    # 화살표
    lines.extend(arrow_line(cx, 150, cx - 140, by + 10, theme))      # x → norm1
    lines.extend(arrow_line(cx - 140, by + 25, cx - 80, by + 25, theme))  # norm → attn
    # residual: x → +
    lines.extend(arrow_line(cx + 60, 125, cx + 130, 125, theme))     # x 옆으로
    # +
    lines.append(f'  <circle cx="{cx + 130}" cy="{by + 25}" r="18" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="2"/>')
    lines.append(f'  <text x="{cx + 130}" y="{by + 31}" text-anchor="middle" font-size="20" font-weight="700" fill="{pal["output"]["text"]}">+</text>')
    # attn → +
    lines.extend(arrow_line(cx + 60, by + 25, cx + 113, by + 25, theme))
    # x (위) → + (residual line)
    lines.append(f'  <path d="M {cx + 130} 125 L {cx + 130} {by + 7}" stroke="{t["arrow"]}" stroke-width="1.8" fill="none" marker-end="url(#arr)" stroke-dasharray="3,3"/>')

    # Block 2: Norm → FFN → +
    by2 = 290
    lines.extend(node(cx - 200, by2, 120, 50, 'gate', theme, title='RMSNorm'))
    lines.extend(node(cx - 60, by2, 120, 50, 'token', theme, title='FFN', sub='SwiGLU'))

    # 위 + → norm2
    lines.extend(arrow_line(cx + 130, by + 43, cx - 140, by2 + 10, theme))
    lines.extend(arrow_line(cx - 140, by2 + 25, cx - 80, by2 + 25, theme))
    # +
    lines.append(f'  <circle cx="{cx + 130}" cy="{by2 + 25}" r="18" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="2"/>')
    lines.append(f'  <text x="{cx + 130}" y="{by2 + 31}" text-anchor="middle" font-size="20" font-weight="700" fill="{pal["output"]["text"]}">+</text>')
    lines.extend(arrow_line(cx + 60, by2 + 25, cx + 113, by2 + 25, theme))
    # 위 + → 아래 + residual
    lines.append(f'  <path d="M {cx + 148} {by + 25} L {cx + 200} {by + 25} L {cx + 200} {by2 + 25} L {cx + 148} {by2 + 25}" stroke="{t["arrow"]}" stroke-width="1.8" fill="none" marker-end="url(#arr)" stroke-dasharray="3,3"/>')

    # 출력
    lines.extend(arrow_line(cx + 130, by2 + 43, cx, 410, theme))
    lines.extend(node(cx - 80, 410, 160, 60, 'output', theme, title='out (B,T,D)', sub='다음 block 으로'))

    # 라벨
    lines.append(f'  <text x="80" y="200" font-size="11" font-weight="700" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">Block 1</text>')
    lines.append(f'  <text x="80" y="305" font-size="11" font-weight="700" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">Block 2</text>')
    lines.append(f'  <text x="700" y="220" font-size="10" fill="{t["legend_text"]}" font-family="JetBrains Mono, monospace">--- residual</text>')

    lines.extend(text_subtitle(cx, 500, 'GPTMini = Block × N (보통 N=6~12) + final RMSNorm + lm_head', theme, size=12))

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('attention-sdpa',        attention_sdpa('light'),        attention_sdpa('dark'))
    save('modern-blocks',         modern_blocks('light'),         modern_blocks('dark'))
    save('memory-stack',          memory_stack('light'),          memory_stack('dark'))
    save('memory-breakdown-pct',  memory_breakdown_pct('light'),  memory_breakdown_pct('dark'))
    save('nanogpt-block',         nanogpt_block('light'),         nanogpt_block('dark'))
    print('Done.')
