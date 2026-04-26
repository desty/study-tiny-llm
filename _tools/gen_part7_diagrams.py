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


# =====================================================================
# Ch 23. 4가지 파인튜닝 길
# =====================================================================

def finetune_paths(theme):
    CW, CH = 1100, 380
    NW, NH = 220, 110
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '4가지 길 — from-scratch · CPT · SFT · LoRA', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '아래로 갈수록 데이터 ↓ · 비용 ↓ · 도메인 적응 능력 ↓', theme))

    rows = [
        ('llm',    'from-scratch',         '100B+ 토큰',      '대규모 GPU 클러스터 · 수일~수주'),
        ('model',  'continued pre-training', '1B~10B 토큰',     '큰 GPU 한 장+ · 하루~수일'),
        ('gate',   'full SFT',             '10K~1M 페어',     'A100 한 장 · 수시간'),
        ('output', 'LoRA / QLoRA',         '100~10K 페어',    '노트북 · 수십 분~수시간'),
    ]
    pal = P(theme); t = T(theme)

    y = 105
    gap = 12
    for role, name, data, cost in rows:
        x = (CW - NW * 3 - 60) // 2
        # 명칭
        lines.extend(node(x, y, NW, NH - 20, role, theme, title=name))
        # 데이터 양
        lines.extend(node(x + NW + 30, y, NW, NH - 20, role, theme, title=data, sub='데이터'))
        # 비용
        lines.extend(node(x + 2 * (NW + 30), y, NW, NH - 20, role, theme, title=cost, sub='비용·시간'))
        y += NH - 20 + gap

    lines.extend(text_subtitle(CW // 2, 350, '본 책 본문 = 1번 (Part 1~6) · 캡스톤 = 4번 (Part 7)', theme, size=12))
    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 28. 3가지 아키텍처
# =====================================================================

def three_architectures(theme):
    CW, CH = 1100, 460
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '3가지 모양 — Encoder · Decoder · Seq2seq', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '같은 트랜스포머에서 마스크 차이로 갈리는 세 형태', theme))

    pal = P(theme); t = T(theme)

    # 3 카드
    cards = [
        ('input',  'Encoder-only', 'BERT / KoELECTRA',
         '양방향 attention\n(mask 없음)', '분류 · NER · 임베딩'),
        ('llm',    'Decoder-only', 'GPT / Llama / Qwen / 본 책',
         'causal mask\n(과거만)', '생성 · 챗봇'),
        ('gate',   'Encoder-Decoder', 'T5 / byT5 / mT5',
         'encoder + cross-attn + decoder', '번역 · 요약 · ITN'),
    ]

    n = 3
    NW, NH = 280, 200
    gap = 30
    total = n * NW + (n - 1) * gap
    left = (CW - total) // 2
    top = 100

    for i, (role, title, models, mask, use) in enumerate(cards):
        x = left + i * (NW + gap)
        # 카드
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="12" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        # 제목
        lines.append(f'  <text x="{x + NW//2}" y="{top + 30}" text-anchor="middle" font-size="16" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 50}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{models}</text>')

        # 작은 attention pattern 그림 (5x5 grid)
        gx = x + NW//2 - 60
        gy = top + 70
        cell = 16
        for r in range(5):
            for c in range(5):
                if role == 'input':
                    fill = pal[role]['stroke']  # 모두 attended (양방향)
                elif role == 'llm':
                    fill = pal[role]['stroke'] if c <= r else 'none'  # causal
                else:
                    fill = pal[role]['stroke'] if (i == 2 and (r >= 3 and c <= r) or (i == 2 and r < 3)) else 'none'
                opa = 0.3 if fill != 'none' else 0
                if fill != 'none':
                    lines.append(f'  <rect x="{gx + c*cell}" y="{gy + r*cell}" width="{cell-1}" height="{cell-1}" fill="{fill}" opacity="0.5"/>')
                lines.append(f'  <rect x="{gx + c*cell}" y="{gy + r*cell}" width="{cell-1}" height="{cell-1}" fill="none" stroke="{t["legend_border"]}" stroke-width="0.4"/>')

        # mask 설명
        for j, line in enumerate(mask.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 175 + j*14}" text-anchor="middle" font-size="10" fill="{pal[role]["sub"]}" font-family="JetBrains Mono, monospace">{line}</text>')

        # 용도
        lines.append(f'  <text x="{x + NW//2}" y="{top + NH + 24}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal[role]["text"]}">{use}</text>')

    lines.extend(text_subtitle(CW // 2, 410, '본 책 본문 = decoder-only · Ch 25 = encoder · Ch 28 = seq2seq', theme, size=12))
    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 22. 5 sLLM 비교 매트릭스
# =====================================================================

def slm_compare_matrix(theme):
    CW, CH = 1100, 480
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '5 sLLM 비교 — 한국어 · 라이선스 · 도구호출 · context', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '본인 작업에 맞는 결정 표 — 2026 기준', theme))

    pal = P(theme); t = T(theme)

    # 표 위치
    headers = ['모델', '크기', '라이선스', '한국어', '도구호출', 'context']
    rows = [
        ('Phi-3.5-mini',  '3.8B',          'MIT',         '△', '◎', '128K'),
        ('SmolLM2',        '0.135~1.7B',   'Apache 2.0',  '×', '△', '8K'),
        ('Gemma 2-2B',     '2B',           'Gemma',       '△', '△', '8K'),
        ('Qwen 2.5',       '0.5~7B',       'Apache 2.0',  '◎', '◎', '32~128K'),
        ('Llama 3.2',      '1~3B',         'Llama 3.2',   '△', '◎', '128K'),
    ]

    # 셀 크기
    col_widths = [180, 130, 160, 130, 130, 160]
    row_h = 50
    table_x = (CW - sum(col_widths)) // 2
    table_y = 100

    # 헤더
    cx = table_x
    for i, h in enumerate(headers):
        lines.append(f'  <rect x="{cx}" y="{table_y}" width="{col_widths[i]}" height="{row_h}" fill="{pal["model"]["fill"]}" stroke="{pal["model"]["stroke"]}" stroke-width="1.5"/>')
        lines.append(f'  <text x="{cx + col_widths[i]//2}" y="{table_y + row_h//2 + 5}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal["model"]["text"]}">{h}</text>')
        cx += col_widths[i]

    # 데이터 행
    score_color = {'◎': pal['output']['stroke'], '○': pal['token']['stroke'],
                    '△': pal['gate']['stroke'], '×': pal['error']['stroke']}

    for r, row in enumerate(rows):
        cx = table_x
        ry = table_y + row_h + r * row_h
        # 모델 이름은 강조
        is_qwen = (row[0] == 'Qwen 2.5')
        bg = pal['output']['fill'] if is_qwen else t['bg']

        for i, val in enumerate(row):
            lines.append(f'  <rect x="{cx}" y="{ry}" width="{col_widths[i]}" height="{row_h}" fill="{bg}" stroke="{t["legend_border"]}" stroke-width="1"/>')
            color = score_color.get(val, t['title'])
            weight = '700' if (val in '◎○△×' or is_qwen) else '500'
            size = 18 if val in '◎○△×' else 12
            lines.append(f'  <text x="{cx + col_widths[i]//2}" y="{ry + row_h//2 + 6}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}">{val}</text>')
            cx += col_widths[i]

    # 추천
    lines.extend(text_subtitle(CW // 2, 410, '한국어 + 라이선스 + 도구호출 → Qwen 2.5 (강조 행)', theme, size=12))
    lines.extend(text_subtitle(CW // 2, 432, '본 책 캡스톤 = Qwen 2.5-0.5B-Instruct + LoRA', theme, size=11))

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 27. Distillation 흐름
# =====================================================================

def distillation_flow(theme):
    CW, CH = 1100, 360
    NW, NH = 160, 80
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'Distillation — Teacher 가 Student 를 가르친다', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '필터가 통과율 50%+ 가 안전선 · Teacher 환각·편향 차단', theme))

    pal = P(theme); t = T(theme)
    y = 130

    stages = [
        ('input',  'Unlabeled\nqueries', '5K~50K', None),
        ('llm',    'Teacher\n추론',      'Qwen 1.5B', None),
        ('error',  'Filter',             'judge + 규칙', '50%+'),
        ('token',  'Student SFT',        'Qwen 0.5B + LoRA', None),
        ('output', 'Deploy',             '3× 빠름·1/3 메모리', None),
    ]

    n = len(stages)
    gap = 30
    total = n * NW + (n - 1) * gap
    left = (CW - total) // 2

    cy = y + NH // 2
    for i in range(n - 1):
        x1 = left + i * (NW + gap) + NW + 2
        x2 = left + (i + 1) * (NW + gap) - 2
        kind = 'warning' if i == 1 else 'primary'  # filter 입구는 warning
        lines.extend(arrow_line(x1, cy, x2, cy, theme, kind=kind))

    for i, (role, title, sub, note) in enumerate(stages):
        x = left + i * (NW + gap)
        # title 에 \n 처리 — 두 줄로
        title_lines = title.split('\n')
        # node helper 가 multiline 안 되니 직접
        h = NH + (20 if note else 10)
        lines.append(f'  <rect x="{x}" y="{y}" width="{NW}" height="{h}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
        for j, tl in enumerate(title_lines):
            lines.append(f'  <text x="{x + NW//2}" y="{y + 28 + j*16}" text-anchor="middle" font-size="14" font-weight="700" fill="{pal[role]["text"]}">{tl}</text>')
        sub_y = y + 28 + len(title_lines) * 16 + 6
        lines.append(f'  <text x="{x + NW//2}" y="{sub_y}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{sub}</text>')
        if note:
            lines.append(f'  <text x="{x + NW//2}" y="{sub_y + 18}" text-anchor="middle" font-size="11" font-weight="700" fill="{pal["output"]["sub"]}">통과 {note}</text>')

    # 캡션
    lines.extend(text_subtitle(CW // 2, 290, 'Gemma 2-2B / SmolLM2 / Phi-3.5 가 모두 이 길로 학습', theme, size=12))
    lines.extend(text_subtitle(CW // 2, 318, '본 책: Qwen 1.5B → 0.5B 미니 distillation 으로 같은 패턴 체험', theme, size=11))

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('lora-structure',       lora_structure('light'),       lora_structure('dark'))
    save('finetune-paths',       finetune_paths('light'),       finetune_paths('dark'))
    save('three-architectures',  three_architectures('light'),  three_architectures('dark'))
    save('slm-compare-matrix',   slm_compare_matrix('light'),   slm_compare_matrix('dark'))
    save('distillation-flow',    distillation_flow('light'),    distillation_flow('dark'))
    print('Done.')
