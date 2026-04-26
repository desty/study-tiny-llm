"""Final SVG diagrams — Ch 3/7/13/15/26/29/30/31."""
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
# Ch 3. 학습 가능 3축 — 메모리·연산·시간
# =====================================================================

def training_budget_axes(theme):
    CW, CH = 1100, 380
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '학습 가능 3축 — 메모리 · 연산 · 시간', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '셋 모두 통과해야 "가능" — 하나라도 실패하면 설계 변경', theme))

    pal = P(theme); t = T(theme)

    axes = [
        ('error',  '메모리',  'OOM\n= Out of Memory',
         '모델 파라미터 + 옵티마이저 상태\n+ activation + gradient',
         '10M fp16 = 20MB\nAdamW 상태 ×3 = 60MB\nactivation ∝ batch·seq'),
        ('gate',   '연산 (FLOPs)', '끝나지 않는 학습',
         '학습 FLOPs ≈ 6 × N × D\nN=파라미터수, D=토큰수',
         '10M × 200M토큰\n≈ 1.2×10¹⁵ FLOPs\nT4=~10¹³ FLOP/s → ~2h'),
        ('token',  '시간',   '일정 초과',
         '메모리 × 연산의 함수\n단위 시간 처리량 =\ntokens/sec',
         '노트북: 5K tok/s\n4h 학습: ~72M 토큰\n≈ TinyStories 1/3'),
    ]

    NW, NH = 280, 160
    gap = 30
    total = len(axes) * NW + (len(axes) - 1) * gap
    left = (CW - total) // 2
    top = 88

    for i, (role, title, fail, formula, numbers) in enumerate(axes):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 22}" text-anchor="middle" font-size="16" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        # fail signal
        lines.append(f'  <text x="{x + NW//2}" y="{top + 38}" text-anchor="middle" font-size="10" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{fail}</text>')
        lines.append(f'  <line x1="{x+16}" y1="{top+44}" x2="{x+NW-16}" y2="{top+44}" stroke="{pal[role]["stroke"]}" stroke-width="0.6" stroke-dasharray="4,3"/>')
        for j, fl in enumerate(formula.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 62 + j*14}" text-anchor="middle" font-size="11" fill="{pal[role]["text"]}">{fl}</text>')
        for j, nl in enumerate(numbers.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 112 + j*14}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{nl}</text>')

    # 아래 결론
    bot_y = top + NH + 28
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="40" rx="8" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 16}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal["output"]["text"]}">본 책 기준선: 10M · 200M 토큰 · M1/M2 또는 Colab T4 · 약 4시간</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 32}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal["output"]["sub"]}">세 축 동시 검사 → 사전에 산수 한 번 → OOM 12시간 손실 방지</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 7. 데이터 품질 4축
# =====================================================================

def data_quality_axes(theme):
    CW, CH = 1100, 360
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '데이터 품질 4축 — 다양성 · 밀도 · 정확성 · 무중복', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '"좋은 학습 데이터" = 4축 모두 합격선 위 — 하나라도 빠지면 깨진다', theme))

    pal = P(theme); t = T(theme)

    axes = [
        ('input',  '다양성\nDiversity',   '모델이 한 패턴만 학습',   'exact dedup\n+ near-dup LSH',  '어휘·구조·스타일이 넓음'),
        ('token',  '밀도\nDensity',       '광고·boilerplate 학습',   'perplexity 필터\n+ classifier', '단위 토큰당 정보량 큼'),
        ('gate',   '정확성\nCorrectness', '환각·오류 학습',           'fact-check 규칙\n+ LLM judge',  '사실·문법이 맞음'),
        ('model',  '무중복\nDedup',       '같은 곳 반복 학습',        'MinHash LSH\n+ URL 중복 제거', '같은 내용 반복 없음'),
    ]

    NW, NH = 220, 140
    gap = 20
    total = len(axes) * NW + (len(axes) - 1) * gap
    left = (CW - total) // 2
    top = 88

    for i, (role, title, fail, tool, desc) in enumerate(axes):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        for j, tl in enumerate(title.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 20 + j*16}" text-anchor="middle" font-size="14" font-weight="700" fill="{pal[role]["text"]}">{tl}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 58}" text-anchor="middle" font-size="10" fill="{pal[role]["sub"]}">{desc}</text>')
        lines.append(f'  <line x1="{x+14}" y1="{top+68}" x2="{x+NW-14}" y2="{top+68}" stroke="{pal[role]["stroke"]}" stroke-width="0.5" stroke-dasharray="3,3"/>')
        # 실패 시
        lines.append(f'  <text x="{x + NW//2}" y="{top + 82}" text-anchor="middle" font-size="10" font-weight="700" fill="{pal["error"]["sub"]}">▲ {fail}</text>')
        # 도구
        for j, tl in enumerate(tool.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 102 + j*14}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{tl}</text>')

    # 아래
    bot_y = top + NH + 26
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="40" rx="8" fill="{pal["llm"]["fill"]}" stroke="{pal["llm"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 16}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal["llm"]["text"]}">Phi-1 의 증명 — 교과서 품질 1B 토큰 > 일반 100B 토큰</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 32}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal["llm"]["sub"]}">데이터 품질 향상이 데이터 양 증가보다 먼저 — 10배 효율 차이</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 13. Mixed Precision — bf16/fp16/fp32
# =====================================================================

def mixed_precision_flow(theme):
    CW, CH = 1100, 380
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'Mixed Precision + Gradient Accumulation', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'bf16 forward · fp32 master copy · 작은 batch × 누적 = 큰 batch 효과', theme))

    pal = P(theme); t = T(theme)

    # 왼쪽: mixed precision 흐름
    mp_left = 60
    mp_top = 86

    fmt_items = [
        ('input',  'fp32 Master Weight', '4 bytes · 정밀한 업데이트\n학습률·모멘텀 상태 보관'),
        ('model',  'bf16 Forward/Backward', '2 bytes · overflow 없음\n(=fp32 지수 범위)\nA100/H100 네이티브 지원'),
        ('gate',   'fp16 (T4 한정)', '2 bytes · overflow 위험\nLoss scaling 필수\nT4 bf16 미지원'),
    ]

    fw = 300; fh = 74; fgap = 14
    for i, (role, title, desc) in enumerate(fmt_items):
        fy = mp_top + i * (fh + fgap)
        lines.append(f'  <rect x="{mp_left}" y="{fy}" width="{fw}" height="{fh}" rx="8" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.8"/>')
        lines.append(f'  <text x="{mp_left + fw//2}" y="{fy + 20}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        for j, dl in enumerate(desc.split('\n')):
            lines.append(f'  <text x="{mp_left + fw//2}" y="{fy + 38 + j*14}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{dl}</text>')

    lines.extend(text_subtitle(mp_left + fw//2, mp_top + 3 * (fh + fgap) + 10, 'autocast(device_type, dtype=torch.bfloat16)', theme, size=10))

    # 오른쪽: gradient accumulation
    ga_left = 460
    ga_top = 86

    lines.append(f'  <text x="{ga_left + 280}" y="{ga_top - 10}" text-anchor="middle" font-size="13" font-weight="700" fill="{t["title"]}">Gradient Accumulation</text>')
    lines.append(f'  <text x="{ga_left + 280}" y="{ga_top + 6}" text-anchor="middle" font-size="10" fill="{t["subtitle"]}">batch=4, accum=8 → effective batch=32</text>')

    # 작은 batch × 8
    bs = 60; bh_ba = 44; bgap2 = 10
    for i in range(8):
        bx = ga_left + i * (bs + bgap2)
        role = 'input' if i < 7 else 'output'
        label = f'mini\nbatch {i+1}' if i < 7 else 'optimizer\n.step()'
        lines.append(f'  <rect x="{bx}" y="{ga_top + 22}" width="{bs}" height="{bh_ba}" rx="5" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
        for j, ll in enumerate(label.split('\n')):
            lines.append(f'  <text x="{bx + bs//2}" y="{ga_top + 36 + j*13}" text-anchor="middle" font-size="9" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{ll}</text>')

    # 화살표 아래
    for i in range(7):
        ax = ga_left + i * (bs + bgap2) + bs + 1
        ay = ga_top + 22 + bh_ba // 2
        lines.extend(arrow_line(ax, ay, ax + bgap2 - 1, ay, theme))

    # 누적 gradient bar
    grad_y = ga_top + 22 + bh_ba + 20
    lines.append(f'  <text x="{ga_left + 280}" y="{grad_y - 4}" text-anchor="middle" font-size="10" fill="{t["subtitle"]}">gradient 누적 (÷8 안 하면 loss scale 주의)</text>')
    for i in range(8):
        bx = ga_left + i * (bs + bgap2)
        filled_w = int((bs - 2) * (i + 1) / 8)
        lines.append(f'  <rect x="{bx}" y="{grad_y}" width="{bs}" height="18" rx="3" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="0.8"/>')
        lines.append(f'  <rect x="{bx}" y="{grad_y}" width="{filled_w}" height="18" rx="3" fill="{pal["token"]["stroke"]}" opacity="0.6"/>')

    ga_note_y = grad_y + 32
    lines.append(f'  <rect x="{ga_left}" y="{ga_note_y}" width="560" height="36" rx="6" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="0.8"/>')
    lines.append(f'  <text x="{ga_left + 280}" y="{ga_note_y + 14}" text-anchor="middle" font-size="11" font-weight="700" fill="{t["legend_text"]}">Gradient Checkpointing: activation 재계산 → 메모리 ↓ 30~40% / 속도 ↓ 10~20%</text>')
    lines.append(f'  <text x="{ga_left + 280}" y="{ga_note_y + 28}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">model.gradient_checkpointing_enable() 한 줄</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 26. 도메인 LoRA — CPT → SFT 두 단계
# =====================================================================

def domain_lora_stages(theme):
    CW, CH = 1100, 360
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '두 단계 도메인 적응 — CPT (선택) + Domain SFT (LoRA)', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '도메인 어휘·스타일 먼저 (CPT) → 도메인 task 형식 학습 (SFT)', theme))

    pal = P(theme); t = T(theme)

    stages = [
        ('model',  '기반 모델',         'Qwen 2.5-0.5B-Instruct\n공개 오픈웨이트',         '이미 한국어 · 도구 호출'),
        ('input',  '① Continued\nPre-training',  'raw text 1B+ 토큰\n도메인 어휘·스타일',  '선택 — 도메인 어휘\n희귀하면 필요'),
        ('gate',   '② Domain SFT\n(LoRA)',        'instruction 페어\n1K~100K',               '필수 — task 형식\n빠름·저비용'),
        ('token',  '어댑터 저장',        'adapter_model.safetensors\n~20MB',                 'merge_and_unload()\n→ GGUF'),
        ('output', '배포',              'GGUF Q4_K_M\n5MB',                                  '노트북·사내 서버\nllama.cpp'),
    ]

    NW, NH = 160, 105
    gap = 28
    total = len(stages) * NW + (len(stages) - 1) * gap
    left = (CW - total) // 2
    top = 90

    for i, (role, title, sub, note) in enumerate(stages):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        for j, tl in enumerate(title.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 18 + j*14}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal[role]["text"]}">{tl}</text>')
        for j, sl in enumerate(sub.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 50 + j*12}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{sl}</text>')
        for j, nl in enumerate(note.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 82 + j*12}" text-anchor="middle" font-size="10" font-weight="700" fill="{pal[role]["text"]}">{nl}</text>')

        if i < len(stages) - 1:
            ax1 = x + NW + 2
            ax2 = x + NW + gap - 2
            ay = top + NH // 2
            kind = 'async' if i == 1 else 'primary'  # CPT 선택 = 점선
            lines.extend(arrow_line(ax1, ay, ax2, ay, theme, kind=kind))

    # 선택/필수 레이블
    opt_x = left + NW + gap // 2
    opt_y = top - 20
    lines.append(f'  <text x="{opt_x}" y="{opt_y}" text-anchor="middle" font-size="10" fill="{pal["input"]["sub"]}">선택 (optional)</text>')
    req_x = left + (NW + gap) * 2 + NW // 2
    lines.append(f'  <text x="{req_x}" y="{opt_y}" text-anchor="middle" font-size="10" font-weight="700" fill="{pal["gate"]["sub"]}">필수</text>')

    # 본 책 캡스톤 설명
    bot_y = top + NH + 28
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="40" rx="8" fill="{pal["gate"]["fill"]}" stroke="{pal["gate"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 16}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal["gate"]["text"]}">본 책 캡스톤 = ② 만 (CPT 스킵) — Qwen 한국어 능력 의존, LoRA 페어만</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 32}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal["gate"]["sub"]}">한국 동화 instruction 페어 ~1K + QLoRA → adapter → GGUF → HF Hub</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 29. 운영 데이터 파이프라인 — PII·라벨·버전
# =====================================================================

def ops_data_pipeline(theme):
    CW, CH = 1100, 360
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '운영 데이터 파이프라인 — PII · 라벨 · 버전', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '실로그는 법무·보안·품질 세 게이트를 통과해야 학습 가능', theme))

    pal = P(theme); t = T(theme)

    stages = [
        ('error',  '원시 실로그',     '콜·채팅·CRM',      'PII 무더기\n개인정보 포함'),
        ('gate',   'PII 마스킹',     '4단계 필터',       '정규식→NER\n→사람→감사'),
        ('input',  '합성 라벨',      'Teacher + 사람',   'LLM 제안\n+ 검수·IAA'),
        ('model',  '데이터 버전',    'DVC/MLflow',       'hash 기록\n학습 1:1 추적'),
        ('output', '학습 준비 완료', '배포 허가',        '법무 승인\n품질 게이트 통과'),
    ]

    NW, NH = 168, 105
    gap = 26
    total = len(stages) * NW + (len(stages) - 1) * gap
    left = (CW - total) // 2
    top = 90

    for i, (role, title, sub, detail) in enumerate(stages):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 20}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 36}" text-anchor="middle" font-size="11" fill="{pal[role]["sub"]}">{sub}</text>')
        lines.append(f'  <line x1="{x+14}" y1="{top+44}" x2="{x+NW-14}" y2="{top+44}" stroke="{pal[role]["stroke"]}" stroke-width="0.5" stroke-dasharray="3,3"/>')
        for j, dl in enumerate(detail.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 60 + j*16}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{dl}</text>')

        if i < len(stages) - 1:
            ax1 = x + NW + 2
            ax2 = x + NW + gap - 2
            ay = top + NH // 2
            lines.extend(arrow_line(ax1, ay, ax2, ay, theme))

    # IAA 설명
    bot_y = top + NH + 28
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="40" rx="8" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="1"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 15}" text-anchor="middle" font-size="11" font-weight="700" fill="{t["legend_text"]}">IAA (Inter-Annotator Agreement) — Cohen κ ≥ 0.8 기준 · 불일치 = 재학습 신호</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 31}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">데이터 버전 = 모델 버전 — 어떤 데이터로 어떤 모델이 나왔는지 1:1 추적</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 30. 회귀·A/B 평가 5축
# =====================================================================

def eval_axes(theme):
    CW, CH = 1100, 380
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '운영 평가 5축 — 회귀 · OOD · Adversarial · A/B · Hold-out', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '배포 전 5축 모두 통과 — 회귀 0% 필수, A/B 는 트래픽 신호', theme))

    pal = P(theme); t = T(theme)

    axes_data = [
        ('model',  'Hold-out\n(Ch 16~18)',  'PPL · 벤치마크\n생성 샘플',    '학습 분포 내\n능력 측정',   '베이스라인'),
        ('input',  '회귀셋',               '이전 버전\n100~500건',         '0% 회귀 필수\n1개라도 깨지면 차단', '배포 게이트'),
        ('gate',   'OOD',                  '학습 외 분포\n새로운 패턴',     '일반화 능력\n측정',         '경고 기준'),
        ('error',  'Adversarial',          '의도적 함정\n공격 패턴',        '안전성·강건성\n측정',        '보안 게이트'),
        ('token',  'A/B 테스트',           '실 트래픽 5~10%\n분기',         '사용자 신호\n만족도',        '최종 확정'),
    ]

    NW, NH = 172, 148
    gap = 18
    total = len(axes_data) * NW + (len(axes_data) - 1) * gap
    left = (CW - total) // 2
    top = 88

    for i, (role, title, what, purpose, gate) in enumerate(axes_data):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        for j, tl in enumerate(title.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 18 + j*14}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal[role]["text"]}">{tl}</text>')
        for j, wl in enumerate(what.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 50 + j*14}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{wl}</text>')
        lines.append(f'  <line x1="{x+12}" y1="{top+80}" x2="{x+NW-12}" y2="{top+80}" stroke="{pal[role]["stroke"]}" stroke-width="0.5" stroke-dasharray="3,3"/>')
        for j, pl in enumerate(purpose.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 96 + j*13}" text-anchor="middle" font-size="10" fill="{pal[role]["text"]}">{pl}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + NH - 8}" text-anchor="middle" font-size="10" font-weight="700" fill="{pal[role]["sub"]}">{gate}</text>')

    lines.extend(text_subtitle(CW // 2, CH - 20, '회귀 = 하드 블로커 | OOD·Adversarial = 경고 | A/B = 트래픽 신호 | Hold-out = 능력 측정', theme, size=11))
    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 31. 서빙 스택 비교
# =====================================================================

def serving_stacks(theme):
    CW, CH = 1100, 380
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '서빙 스택 비교 — llama.cpp · vLLM · TGI · Ollama', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '같은 모델, 다른 껍데기 — 처리량·환경·운영 복잡도가 갈림', theme))

    pal = P(theme); t = T(theme)

    stacks = [
        ('model',  'llama.cpp\nserver',      'CPU/Mac/Vulkan',  '사내·소규모\n동시 ≤ 10',
         '가벼움 · 설치 쉬움',          'vLLM 대비 처리량 낮음'),
        ('llm',    'vLLM',               'GPU 필수',        'GPU 1장+\n많은 동시 사용자',
         'PagedAttention\n처리량 최강',   '무거움 · GPU 전용'),
        ('input',  'HF TGI',             'GPU 권장',        'HF 생태계\n의존 환경',
         '표준 HF 모델 자동',           '운영 학습곡선'),
        ('gate',   'Ollama',             'Mac/Linux/Win',   '데모·개발',
         '사용자 친화\n설정 최소',       '운영급 X'),
    ]

    NW, NH = 220, 170
    gap = 20
    total = len(stacks) * NW + (len(stacks) - 1) * gap
    left = (CW - total) // 2
    top = 86

    for i, (role, title, hw, usecase, pros, cons) in enumerate(stacks):
        x = left + i * (NW + gap)
        is_book = (title in ('llama.cpp\nserver', 'vLLM'))
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="{"2.5" if is_book else "1.5"}"/>')
        for j, tl in enumerate(title.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 20 + j*16}" text-anchor="middle" font-size="15" font-weight="700" fill="{pal[role]["text"]}">{tl}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 54}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{hw}</text>')
        lines.append(f'  <line x1="{x+14}" y1="{top+62}" x2="{x+NW-14}" y2="{top+62}" stroke="{pal[role]["stroke"]}" stroke-width="0.6" stroke-dasharray="4,3"/>')
        for j, ul in enumerate(usecase.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 76 + j*14}" text-anchor="middle" font-size="11" fill="{pal[role]["text"]}">{ul}</text>')
        lines.append(f'  <line x1="{x+14}" y1="{top+106}" x2="{x+NW-14}" y2="{top+106}" stroke="{pal[role]["stroke"]}" stroke-width="0.4" stroke-dasharray="3,3"/>')
        for j, pl in enumerate(pros.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 120 + j*13}" text-anchor="middle" font-size="10" fill="{pal[role]["text"]}">{pl}</text>')
        for j, cl in enumerate(cons.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 148 + j*13}" text-anchor="middle" font-size="10" fill="{pal[role]["sub"]}">{cl}</text>')
        if is_book:
            lines.append(f'  <text x="{x + NW//2}" y="{top - 10}" text-anchor="middle" font-size="10" font-weight="700" fill="{pal[role]["sub"]}">본 책 권장</text>')

    lines.extend(text_subtitle(CW // 2, CH - 20, '동시 ≤ 10 사내 → llama.cpp server · GPU 많은 사용자 → vLLM', theme, size=12))
    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 15. 4시간 훈련 실전 — 조각 모음
# =====================================================================

def four_hour_run(theme):
    CW, CH = 1100, 360
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '4시간 훈련 실전 — 모든 조각이 합쳐지는 순간', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'Ch 5~14 조각 6개 → 한 번의 학습 실행 → final.pt', theme))

    pal = P(theme); t = T(theme)

    # 왼쪽: 조각 6개
    pieces = [
        ('input',  'Ch 5  데이터',    'TinyStories 200M 토큰'),
        ('token',  'Ch 6  토크나이저', 'ByteLevel BPE 8K'),
        ('model',  'Ch 10 모델',       'GPTMini 10M (decoder)'),
        ('llm',    'Ch 12 학습루프',   'AdamW + cosine'),
        ('gate',   'Ch 13 정밀도',     'bf16 / fp16'),
        ('memory', 'Ch 14 로깅',       'jsonl + last.pt'),
    ]

    pw, ph, pgap = 270, 34, 8
    px = 60
    py_start = 90

    for i, (role, title, sub) in enumerate(pieces):
        py = py_start + i * (ph + pgap)
        lines.append(f'  <rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="7" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
        lines.append(f'  <text x="{px + 12}" y="{py + 22}" font-size="12" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{px + pw - 10}" y="{py + 22}" text-anchor="end" font-size="10" fill="{pal[role]["sub"]}">{sub}</text>')

    # 화살표 → 중앙 실행
    pieces_bottom = py_start + len(pieces) * (ph + pgap) - pgap
    pieces_mid = (py_start + pieces_bottom) // 2
    run_x = px + pw + 30
    run_w, run_h = 180, 80
    run_y = (CH - run_h) // 2
    lines.extend(arrow_line(px + pw + 2, pieces_mid, run_x - 2, run_y + run_h // 2, theme, kind='success'))

    # 실행 박스
    lines.append(f'  <rect x="{run_x}" y="{run_y}" width="{run_w}" height="{run_h}" rx="10" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="2.5"/>')
    lines.append(f'  <text x="{run_x + run_w//2}" y="{run_y + 28}" text-anchor="middle" font-size="15" font-weight="700" fill="{pal["output"]["text"]}">학습 실행</text>')
    lines.append(f'  <text x="{run_x + run_w//2}" y="{run_y + 46}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal["output"]["sub"]}">~4시간</text>')
    lines.append(f'  <text x="{run_x + run_w//2}" y="{run_y + 62}" text-anchor="middle" font-size="10" fill="{pal["output"]["sub"]}">T4 / M1·M2</text>')

    # 오른쪽: 결과
    res_x = run_x + run_w + 30
    results = [
        ('output',  'final.pt',       '체크포인트 · 배포 준비'),
        ('token',   '손실 곡선',      'jsonl → 진단 · 재학습 결정'),
        ('model',   '동화 샘플 5개',  '생성 품질 육안 확인'),
        ('gate',    'PPL 측정',       'Ch 16 → 숫자로 검증'),
    ]

    rw, rh, rgap = 270, 34, 8
    ry_start = (CH - len(results) * (rh + rgap)) // 2 + 10

    lines.extend(arrow_line(run_x + run_w + 2, run_y + run_h // 2, res_x - 2, ry_start + (len(results) * (rh + rgap)) // 2, theme, kind='primary'))

    for i, (role, title, sub) in enumerate(results):
        ry = ry_start + i * (rh + rgap)
        lines.append(f'  <rect x="{res_x}" y="{ry}" width="{rw}" height="{rh}" rx="7" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
        lines.append(f'  <text x="{res_x + 12}" y="{ry + 22}" font-size="12" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{res_x + rw - 10}" y="{ry + 22}" text-anchor="end" font-size="10" fill="{pal[role]["sub"]}">{sub}</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('training-budget-axes', training_budget_axes('light'), training_budget_axes('dark'))
    save('data-quality-axes',    data_quality_axes('light'),    data_quality_axes('dark'))
    save('mixed-precision-flow', mixed_precision_flow('light'), mixed_precision_flow('dark'))
    save('domain-lora-stages',   domain_lora_stages('light'),   domain_lora_stages('dark'))
    save('ops-data-pipeline',    ops_data_pipeline('light'),    ops_data_pipeline('dark'))
    save('eval-axes',            eval_axes('light'),            eval_axes('dark'))
    save('serving-stacks',       serving_stacks('light'),       serving_stacks('dark'))
    save('four-hour-run',        four_hour_run('light'),        four_hour_run('dark'))
    print('Done.')
