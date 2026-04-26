"""Part 2 SVG diagrams — Ch 5 TinyStories · Ch 6 BPE."""
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
# Ch 5. 합성 데이터 3줄기
# =====================================================================

def synth_data_streams(theme):
    CW, CH = 1100, 420
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '합성 데이터 3줄기 — TinyStories · Phi · Cosmopedia', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '작은 모델의 부활을 이끈 세 계보 — 아래로 갈수록 규모 ↑', theme))

    pal = P(theme); t = T(theme)

    streams = [
        ('llm',    'TinyStories 라인',   'Eldan & Li, 2023',   '좁은 도메인\n+ 합성 동화',   '200M 토큰 · 3-4세 어휘\n1M 모델도 일관성'),
        ('model',  'Phi 라인',           'Phi-1, 2023',        '교과서 스타일\n합성 데이터', '코드·추론 압축\n1.3B > 7B on HumanEval'),
        ('token',  'Cosmopedia 라인',    'HuggingFace, 2024',  '대규모 오픈\n합성 코퍼스',  '30B 토큰 · Mixtral 생성\nSmolLM2 학습 데이터'),
    ]

    NW, NH = 240, 110
    gap = 40
    total = len(streams) * NW + (len(streams) - 1) * gap
    left = (CW - total) // 2
    row_y = 90

    for i, (role, name, src, desc, stats) in enumerate(streams):
        x = left + i * (NW + gap)
        # 메인 카드
        lines.append(f'  <rect x="{x}" y="{row_y}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        lines.append(f'  <text x="{x + NW//2}" y="{row_y + 26}" text-anchor="middle" font-size="15" font-weight="700" fill="{pal[role]["text"]}">{name}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{row_y + 44}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{src}</text>')
        for j, line in enumerate(desc.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{row_y + 65 + j*14}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal[role]["text"]}">{line}</text>')

        # stats 카드 아래
        sy = row_y + NH + 16
        lines.append(f'  <rect x="{x}" y="{sy}" width="{NW}" height="64" rx="8" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="1"/>')
        for j, line in enumerate(stats.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{sy + 22 + j*16}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">{line}</text>')

        # 아래 화살표 → 공통 풀
        ay1 = sy + 64
        ay2 = CH - 56
        lines.extend(arrow_line(x + NW//2, ay1, x + NW//2, ay2, theme))

    # 공통 "학습 데이터 풀"
    pw, ph = 400, 46
    px = (CW - pw) // 2
    py = CH - 56
    lines.append(f'  <rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="2"/>')
    lines.append(f'  <text x="{CW//2}" y="{py + 20}" text-anchor="middle" font-size="14" font-weight="700" fill="{pal["output"]["text"]}">학습 데이터 풀</text>')
    lines.append(f'  <text x="{CW//2}" y="{py + 38}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal["output"]["sub"]}">본 책 = TinyStories 영어판 + 한국어 합성본 혼합</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 6. BPE 합치기 단계
# =====================================================================

def bpe_merge_steps(theme):
    CW, CH = 1100, 400
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'BPE — 가장 자주 나오는 쌍을 반복해서 합친다', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '초기: 글자 단위 → 반복: 빈도 1위 쌍 합치기 → 종료: vocab size 도달', theme))

    pal = P(theme); t = T(theme)

    # Step columns
    steps = [
        ('Step 0\n초기화',    ['l', 'o', 'w', '⟨/w⟩'],                  '"low" 를 글자 단위로'),
        ('Step 1\n(l o→lo)',  ['lo', 'w', '⟨/w⟩'],                      'l·o 가 가장 자주 붙어 등장'),
        ('Step 2\n(lo w→low)', ['low', '⟨/w⟩'],                         'lo·w 다음으로 자주'),
        ('Step 3\n(low ⟨/w⟩→low⟨/w⟩)', ['low⟨/w⟩'],                    '단어 경계 합치기 완료'),
    ]

    NW_step = 200
    gap_step = 60
    total = len(steps) * NW_step + (len(steps) - 1) * gap_step
    left = (CW - total) // 2
    top = 90

    for i, (step_label, tokens, caption) in enumerate(steps):
        sx = left + i * (NW_step + gap_step)
        # 단계 헤더
        for j, sl in enumerate(step_label.split('\n')):
            lines.append(f'  <text x="{sx + NW_step//2}" y="{top + j*16}" text-anchor="middle" font-size="12" font-weight="700" fill="{t["title"]}">{sl}</text>')

        # 토큰 칩들
        token_y = top + 42
        chip_gap = 8
        chip_h = 32
        for tok in tokens:
            chip_w = max(40, len(tok) * 11 + 16)
            chip_x = sx + (NW_step - chip_w) // 2
            role = 'token' if len(tok) > 2 else 'input'
            lines.append(f'  <rect x="{chip_x}" y="{token_y}" width="{chip_w}" height="{chip_h}" rx="6" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
            lines.append(f'  <text x="{chip_x + chip_w//2}" y="{token_y + 21}" text-anchor="middle" font-size="13" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{tok}</text>')
            token_y += chip_h + chip_gap

        # 아래 캡션
        lines.append(f'  <text x="{sx + NW_step//2}" y="{CH - 50}" text-anchor="middle" font-size="10" fill="{t["subtitle"]}">{caption}</text>')

        # 화살표 to next
        if i < len(steps) - 1:
            ax1 = sx + NW_step + 2
            ax2 = sx + NW_step + gap_step - 2
            ay = top + 76
            lines.extend(arrow_line(ax1, ay, ax2, ay, theme, kind='success'))
            # 합치기 레이블
            mx = sx + NW_step + gap_step // 2
            lines.append(f'  <text x="{mx}" y="{ay - 8}" text-anchor="middle" font-size="9" font-family="JetBrains Mono, monospace" fill="{t["subtitle"]}">merge</text>')

    # 아래: vocab 성장 요약
    bot_y = CH - 28
    lines.append(f'  <text x="{CW//2}" y="{bot_y}" text-anchor="middle" font-size="12" font-weight="700" fill="{t["legend_text"]}">실제 BPE: 이 과정을 수만 번 → vocab size (본 책 8K) 까지</text>')

    # vocab 성장 bar (오른쪽 아래)
    bar_x = CW - 280
    bar_y = 220
    bar_labels = [('7', 'init'), ('8', '+1'), ('9', '+2'), ('8K', 'final')]
    bw, bh, bgap = 40, 20, 10
    lines.append(f'  <text x="{bar_x + 90}" y="{bar_y - 10}" text-anchor="middle" font-size="10" font-weight="700" fill="{t["subtitle"]}">vocab 성장</text>')
    for j, (val, lbl) in enumerate(bar_labels):
        bx = bar_x + j * (bw + bgap)
        height_frac = [0.15, 0.20, 0.25, 1.0][j]
        bbar_h = int(70 * height_frac)
        by = bar_y + 70 - bbar_h
        role = 'output' if j == 3 else 'token'
        lines.append(f'  <rect x="{bx}" y="{by}" width="{bw}" height="{bbar_h}" rx="3" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.2"/>')
        lines.append(f'  <text x="{bx + bw//2}" y="{bar_y + 84}" text-anchor="middle" font-size="9" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">{lbl}</text>')
        lines.append(f'  <text x="{bx + bw//2}" y="{by - 3}" text-anchor="middle" font-size="9" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{val}</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('synth-data-streams', synth_data_streams('light'), synth_data_streams('dark'))
    save('bpe-merge-steps',    bpe_merge_steps('light'),    bpe_merge_steps('dark'))
    print('Done.')
