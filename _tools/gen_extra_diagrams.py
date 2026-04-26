"""Extra SVG diagrams — Ch 12 학습루프 · Ch 18 내부시각화 · Ch 20 GGUF · Ch 25 NER."""
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
# Ch 12. 한 step 의 6단계 학습 루프
# =====================================================================

def training_loop_steps(theme):
    CW, CH = 1100, 360
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '한 Step 의 6단계 — 모든 학습 루프의 본체', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'nanoGPT · Llama · GPT-4 모두 동일 — 분산·precision 만 다름', theme))

    pal = P(theme); t = T(theme)

    steps = [
        ('input',  '1\nforward',   'logits, loss\n= model(x, y)',     '현재 파라미터로 예측'),
        ('model',  '2\nbackward',  'loss.backward()',                   'gradient 계산'),
        ('gate',   '3\nclip',      'clip_grad_norm_\n(model, 1.0)',    '발산 방지'),
        ('llm',    '4\nstep',      'optimizer.step()',                  'θ -= lr × grad'),
        ('error',  '5\nzero_grad', 'optimizer\n.zero_grad()',          '다음 step 초기화'),
        ('token',  '6\nscheduler', 'scheduler.step()',                  'lr warmup→cosine'),
    ]

    NW, NH = 140, 108
    gap = 18
    total = len(steps) * NW + (len(steps) - 1) * gap
    left = (CW - total) // 2
    top = 88

    for i, (role, num, code, caption) in enumerate(steps):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        # Step number badge
        lines.append(f'  <circle cx="{x + NW//2}" cy="{top + 18}" r="14" fill="{pal[role]["stroke"]}"/>')
        num_line = num.split('\n')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 14}" text-anchor="middle" font-size="9" font-weight="700" fill="{t["badge_num_fill"]}" font-family="JetBrains Mono, monospace">{num_line[0]}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 24}" text-anchor="middle" font-size="9" font-weight="700" fill="{t["badge_num_fill"]}" font-family="JetBrains Mono, monospace">{num_line[1]}</text>')
        # code
        for j, cl in enumerate(code.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 52 + j*14}" text-anchor="middle" font-size="10.5" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{cl}</text>')
        # caption
        lines.append(f'  <text x="{x + NW//2}" y="{top + NH - 8}" text-anchor="middle" font-size="10" fill="{pal[role]["sub"]}">{caption}</text>')

        # 화살표
        if i < len(steps) - 1:
            ax1 = x + NW + 2
            ax2 = x + NW + gap - 2
            ay = top + NH // 2
            lines.extend(arrow_line(ax1, ay, ax2, ay, theme))

    # 루프 피드백 화살표 (마지막 → 첫 번째)
    loop_y = top + NH + 22
    left_x = left + NW // 2
    right_x = left + (len(steps) - 1) * (NW + gap) + NW // 2
    lines.append(f'  <path d="M {right_x},{top + NH + 2} L {right_x},{loop_y} L {left_x},{loop_y} L {left_x},{top + NH + 2}" stroke="{t["arrow"]}" stroke-width="1.5" fill="none" stroke-dasharray="5,3" marker-end="url(#arr)"/>')
    lines.append(f'  <text x="{CW//2}" y="{loop_y + 16}" text-anchor="middle" font-size="11" fill="{t["subtitle"]}">다음 batch 반복 (수천~수만 step)</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 18. Attention map vs Logit 분포
# =====================================================================

def internal_signals(theme):
    CW, CH = 1100, 420
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '모델 내부 두 신호 — Attention Map · Logit 분포', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '출력이 아닌 내부를 보면 학습 상태·실패 원인이 직접 보인다', theme))

    pal = P(theme); t = T(theme)

    # 왼쪽: Attention Map 예시
    am_x, am_y = 80, 90
    AM = 200
    lines.append(f'  <text x="{am_x + AM//2}" y="{am_y - 10}" text-anchor="middle" font-size="13" font-weight="700" fill="{t["title"]}">Attention Map</text>')
    lines.append(f'  <text x="{am_x + AM//2}" y="{am_y + 6}" text-anchor="middle" font-size="10" fill="{t["subtitle"]}">head × (T×T) softmax 행렬</text>')

    toks = ['Once', 'upon', 'a', 'time', ',']
    cell = AM // len(toks)
    # causal attention pattern (lower triangular, graduated)
    attn_vals = [
        [0.9, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.5, 0.0, 0.0],
        [0.1, 0.2, 0.2, 0.5, 0.0],
        [0.1, 0.1, 0.1, 0.3, 0.4],
    ]
    grid_y = am_y + 20
    for r in range(len(toks)):
        for c in range(len(toks)):
            val = attn_vals[r][c]
            if val > 0:
                # blue intensity
                intensity = int(val * 200)
                if theme == 'light':
                    fill = f'rgb({255-intensity},{255-intensity},255)'
                else:
                    fill = f'rgba(96,165,250,{val:.2f})'
            else:
                fill = t['grid']
            lines.append(f'  <rect x="{am_x + c*cell}" y="{grid_y + r*cell}" width="{cell-1}" height="{cell-1}" rx="2" fill="{fill}" stroke="{t["legend_border"]}" stroke-width="0.5"/>')
            if val > 0:
                lines.append(f'  <text x="{am_x + c*cell + cell//2}" y="{grid_y + r*cell + cell//2 + 4}" text-anchor="middle" font-size="9" fill="{pal["input"]["text"]}">{val:.1f}</text>')
    # 토큰 레이블 (x축)
    for c, tok in enumerate(toks):
        lines.append(f'  <text x="{am_x + c*cell + cell//2}" y="{grid_y - 4}" text-anchor="middle" font-size="8" fill="{t["subtitle"]}">{tok}</text>')
    # 토큰 레이블 (y축)
    for r, tok in enumerate(toks):
        lines.append(f'  <text x="{am_x - 4}" y="{grid_y + r*cell + cell//2 + 4}" text-anchor="end" font-size="8" fill="{t["subtitle"]}">{tok}</text>')

    # 학습 전/후 설명
    cap_y = grid_y + AM + 16
    lines.append(f'  <rect x="{am_x}" y="{cap_y}" width="{AM}" height="52" rx="6" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="0.8"/>')
    caps = ['학습 전: 균등 (uniform)', '학습 후: 특화 (specialized)', ' — head 별로 다른 패턴']
    for j, c in enumerate(caps):
        lines.append(f'  <text x="{am_x + AM//2}" y="{cap_y + 16 + j*14}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">{c}</text>')

    # 오른쪽: Logit 분포 바 차트
    lg_x = 460
    lg_y = 90
    lines.append(f'  <text x="{lg_x + 280}" y="{lg_y - 10}" text-anchor="middle" font-size="13" font-weight="700" fill="{t["title"]}">Logit 분포</text>')
    lines.append(f'  <text x="{lg_x + 280}" y="{lg_y + 6}" text-anchor="middle" font-size="10" fill="{t["subtitle"]}">(vocab,) 마지막 레이어 출력</text>')

    # 정상 분포 vs 비정상
    scenarios = [
        ('token', '정상 분포', [0.35, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.07], '적절히 피크'),
        ('error', '붕괴 (반복)', [0.98, 0.01, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], 'top-1 99%→반복'),
        ('gate', '학습 부족', [0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.12], '균등→헷갈림'),
    ]

    sc_w = 170
    sc_gap = 24
    bar_max_h = 120
    for si, (role, title, vals, note) in enumerate(scenarios):
        sx = lg_x + si * (sc_w + sc_gap)
        lines.append(f'  <text x="{sx + sc_w//2}" y="{lg_y + 22}" text-anchor="middle" font-size="11" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        bw = (sc_w - 20) // len(vals)
        baseline = lg_y + 22 + bar_max_h
        for bi, v in enumerate(vals):
            bh = int(v * bar_max_h)
            bx = sx + 10 + bi * bw
            by = baseline - bh
            lines.append(f'  <rect x="{bx}" y="{by}" width="{bw-2}" height="{bh}" rx="2" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1"/>')
        lines.append(f'  <line x1="{sx+8}" y1="{baseline}" x2="{sx+sc_w-8}" y2="{baseline}" stroke="{t["legend_border"]}" stroke-width="0.8"/>')
        lines.append(f'  <text x="{sx + sc_w//2}" y="{baseline + 18}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{note}</text>')

    # 아래 요약
    bot_y = CH - 46
    lines.append(f'  <rect x="{am_x}" y="{bot_y}" width="{CW - am_x*2}" height="38" rx="8" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 16}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal["output"]["text"]}">사용처: 학습 진단 · 실패 분석 · head 해석 · logit lens</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 32}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal["output"]["sub"]}">작은 모델일수록 직접 시각화가 유일한 디버깅 수단</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 20. GGUF 변환 파이프라인
# =====================================================================

def gguf_pipeline(theme):
    CW, CH = 1100, 360
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'HF → GGUF 변환 파이프라인', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '모델 + 토크나이저 + 메타데이터를 단일 파일로 — llama.cpp 표준 포맷', theme))

    pal = P(theme); t = T(theme)

    stages = [
        ('model',  'HF 체크포인트',      'final.pt',              'PyTorch\nstate_dict'),
        ('gate',   'convert_hf_to_gguf', 'llama.cpp 스크립트',    'FP16 GGUF\n(무손실)'),
        ('llm',    'quantize',           'llama-quantize',         'Q4_K_M GGUF\n(int4 압축)'),
        ('token',  'llama-cli',          'llama-cli -m 모델.gguf', '즉시 추론\n노트북에서'),
    ]

    NW, NH = 190, 100
    gap = 40
    total = len(stages) * NW + (len(stages) - 1) * gap
    left = (CW - total) // 2
    top = 100

    for i, (role, title, cmd, detail) in enumerate(stages):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 22}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 38}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{cmd}</text>')
        for j, dl in enumerate(detail.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 62 + j*14}" text-anchor="middle" font-size="11" font-weight="700" fill="{pal[role]["text"]}">{dl}</text>')

        if i < len(stages) - 1:
            ax1 = x + NW + 2
            ax2 = x + NW + gap - 2
            ay = top + NH // 2
            lines.extend(arrow_line(ax1, ay, ax2, ay, theme, kind='primary'))

    # 파일 크기 비교
    size_y = top + NH + 26
    sizes = [('model', '20 MB\n(fp16)'), ('gate', '20 MB\n(fp16 GGUF)'), ('llm', '5 MB\n(Q4_K_M)'), ('token', '실행')]
    for i, (role, sz) in enumerate(sizes):
        x = left + i * (NW + gap)
        for j, sl in enumerate(sz.split('\n')):
            fw = '700' if j == 0 else '500'
            lines.append(f'  <text x="{x + NW//2}" y="{size_y + j*15}" text-anchor="middle" font-size="11" font-weight="{fw}" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{sl}</text>')

    # GGUF 내용 박스
    gguf_note_y = size_y + 46
    lines.append(f'  <rect x="{left}" y="{gguf_note_y}" width="{total}" height="34" rx="6" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="0.8"/>')
    lines.append(f'  <text x="{CW//2}" y="{gguf_note_y + 14}" text-anchor="middle" font-size="11" font-weight="700" fill="{t["legend_text"]}">GGUF 한 파일 안: magic "GGUF" · metadata · vocab+merges · 양자화 가중치</text>')
    lines.append(f'  <text x="{CW//2}" y="{gguf_note_y + 28}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">config.json + tokenizer.json + state_dict → 단 하나의 .gguf 파일</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 25. Encoder NER 파이프라인
# =====================================================================

def encoder_ner_pipeline(theme):
    CW, CH = 1100, 380
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'Encoder NER — IOB 태깅 파이프라인', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'Decoder (생성) 대신 Encoder (양방향) — 빠름 · 작음 · NER 에 강함', theme))

    pal = P(theme); t = T(theme)

    # 상단: 파이프라인 흐름
    pipe_stages = [
        ('input',  '텍스트 입력',     '"AICC 고객 홍길동\n씨 전화"',    None),
        ('model',  'Encoder\n토크나이저', 'token → id\n양방향 context',  None),
        ('llm',    'Token\nClassifier', '[CLS] hidden\n→ IOB label',     None),
        ('output', 'NER 결과',        'B-PER: 홍길동\nO: 전화',        None),
    ]

    NW, NH = 190, 105
    gap = 36
    total = len(pipe_stages) * NW + (len(pipe_stages) - 1) * gap
    left = (CW - total) // 2
    top = 86

    for i, (role, title, detail, _) in enumerate(pipe_stages):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        for j, tl in enumerate(title.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 20 + j*14}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal[role]["text"]}">{tl}</text>')
        for j, dl in enumerate(detail.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 56 + j*16}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{dl}</text>')

        if i < len(pipe_stages) - 1:
            ax1 = x + NW + 2
            ax2 = x + NW + gap - 2
            ay = top + NH // 2
            lines.extend(arrow_line(ax1, ay, ax2, ay, theme))

    # 아래: IOB 태깅 예시
    iob_y = top + NH + 28
    tokens = ['AICC', '고객', '홍길동', '씨', '전화']
    iob_tags = ['O', 'O', 'B-PER', 'I-PER', 'O']
    iob_roles = ['gate', 'gate', 'input', 'model', 'gate']
    tok_w = 120
    tok_h = 40
    tok_gap = 8
    tok_total = len(tokens) * tok_w + (len(tokens) - 1) * tok_gap
    tok_left = (CW - tok_total) // 2

    lines.append(f'  <text x="{CW//2}" y="{iob_y - 4}" text-anchor="middle" font-size="11" font-weight="700" fill="{t["subtitle"]}">IOB 태깅 예시</text>')

    for i, (tok, tag, role) in enumerate(zip(tokens, iob_tags, iob_roles)):
        tx = tok_left + i * (tok_w + tok_gap)
        # 토큰 박스
        lines.append(f'  <rect x="{tx}" y="{iob_y + 6}" width="{tok_w}" height="30" rx="5" fill="{pal["token"]["fill"]}" stroke="{pal["token"]["stroke"]}" stroke-width="1.2"/>')
        lines.append(f'  <text x="{tx + tok_w//2}" y="{iob_y + 26}" text-anchor="middle" font-size="12" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal["token"]["text"]}">{tok}</text>')
        # 태그 박스
        ty = iob_y + 6 + 30 + 6
        is_ner = tag != 'O'
        t_role = 'input' if tag == 'B-PER' else ('model' if tag == 'I-PER' else 'gate')
        lines.append(f'  <rect x="{tx}" y="{ty}" width="{tok_w}" height="26" rx="5" fill="{pal[t_role]["fill"]}" stroke="{pal[t_role]["stroke"]}" stroke-width="1.2"/>')
        lines.append(f'  <text x="{tx + tok_w//2}" y="{ty + 17}" text-anchor="middle" font-size="11" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[t_role]["text"]}">{tag}</text>')

    # 범례
    leg_y = iob_y + 86
    legend_items = [('input', 'B-PER: 개체명 시작'), ('model', 'I-PER: 개체명 계속'), ('gate', 'O: 개체명 아님')]
    for i, (role, label) in enumerate(legend_items):
        lx = CW//2 - 260 + i * 180
        lines.append(f'  <rect x="{lx}" y="{leg_y}" width="12" height="12" rx="2" fill="{pal[role]["stroke"]}"/>')
        lines.append(f'  <text x="{lx + 18}" y="{leg_y + 10}" font-size="11" fill="{t["legend_text"]}">{label}</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('training-loop-steps',  training_loop_steps('light'),  training_loop_steps('dark'))
    save('internal-signals',     internal_signals('light'),     internal_signals('dark'))
    save('gguf-pipeline',        gguf_pipeline('light'),        gguf_pipeline('dark'))
    save('encoder-ner-pipeline', encoder_ner_pipeline('light'), encoder_ner_pipeline('dark'))
    print('Done.')
