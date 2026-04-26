"""Part 6 SVG diagrams — Ch 19 양자화 · Ch 21 챗봇 데모 형태."""
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
# Ch 19. 양자화 — 비트 압축과 트레이드오프
# =====================================================================

def quant_bit_tradeoff(theme):
    CW, CH = 1100, 420
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '양자화 — 비트 줄이기와 트레이드오프', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '비트 절반 = 메모리 절반 · 정확도 손실은 양자화 품질에 달려 있다', theme))

    pal = P(theme); t = T(theme)

    formats = [
        ('model',  'fp32',  4,    '±3.4×10³⁸',   '기준 (0%)',   '훈련 중 gradient',     1.0),
        ('input',  'fp16',  2,    '±6.5×10⁴',    '<1% 손실',    '추론·혼합 정밀도',     0.5),
        ('token',  'int8',  1,    '-128 ~ 127',   '2~5% 손실',   'llama.cpp Q8_0',      0.25),
        ('gate',   'int4',  0.5,  '-8 ~ 7',       '5~15% 손실',  'GGUF Q4_K_M · 본 책', 0.125),
    ]

    # 왼쪽: 막대 그래프 (메모리)
    bar_left = 80
    bar_top = 100
    bar_max_w = 340
    bar_h = 44
    bar_gap = 18

    lines.append(f'  <text x="{bar_left}" y="{bar_top - 14}" font-size="12" font-weight="700" fill="{t["title"]}">메모리 (10M 모델 기준)</text>')

    for i, (role, fmt, byt, rng, acc, use, frac) in enumerate(formats):
        by = bar_top + i * (bar_h + bar_gap)
        bw = int(bar_max_w * frac)
        # 막대
        lines.append(f'  <rect x="{bar_left}" y="{by}" width="{bw}" height="{bar_h}" rx="6" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="1.5"/>')
        # 포맷 이름
        lines.append(f'  <text x="{bar_left + 12}" y="{by + 27}" font-size="14" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{fmt}</text>')
        # bytes
        lines.append(f'  <text x="{bar_left + 80}" y="{by + 20}" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{byt} bytes</text>')
        # MB 값 (오른쪽)
        mb = byt * 10  # 10M params
        mb_str = f'{mb:.0f}MB' if mb >= 1 else f'{mb*1000:.0f}KB'
        lines.append(f'  <text x="{bar_left + bw + 8}" y="{by + 27}" font-size="12" font-weight="700" font-family="JetBrains Mono, monospace" fill="{pal[role]["text"]}">{mb_str}</text>')

    # 오른쪽: 상세 표
    tbl_left = 520
    tbl_top = 86
    col_ws = [80, 140, 120, 200]
    headers = ['형식', '정확도', 'bytes', '용도']
    row_h = 44

    # 헤더
    cx = tbl_left
    for i, h in enumerate(headers):
        lines.append(f'  <rect x="{cx}" y="{tbl_top}" width="{col_ws[i]}" height="32" fill="{pal["model"]["fill"]}" stroke="{pal["model"]["stroke"]}" stroke-width="1"/>')
        lines.append(f'  <text x="{cx + col_ws[i]//2}" y="{tbl_top + 20}" text-anchor="middle" font-size="12" font-weight="700" fill="{pal["model"]["text"]}">{h}</text>')
        cx += col_ws[i]

    for r, (role, fmt, byt, rng, acc, use, frac) in enumerate(formats):
        cx = tbl_left
        ry = tbl_top + 32 + r * row_h
        is_book = (fmt == 'int4')
        bg = pal['output']['fill'] if is_book else t['bg']
        vals = [fmt, acc, f'{byt}', use]
        for i, val in enumerate(vals):
            lines.append(f'  <rect x="{cx}" y="{ry}" width="{col_ws[i]}" height="{row_h}" fill="{bg}" stroke="{t["legend_border"]}" stroke-width="1"/>')
            weight = '700' if (i == 0 or is_book) else '500'
            color = pal[role]['text'] if i == 0 else (pal['output']['text'] if is_book else t['legend_text'])
            lines.append(f'  <text x="{cx + col_ws[i]//2}" y="{ry + row_h//2 + 5}" text-anchor="middle" font-size="11" font-weight="{weight}" font-family="JetBrains Mono, monospace" fill="{color}">{val}</text>')
            cx += col_ws[i]

    lines.extend(text_subtitle(CW // 2, CH - 20, '본 책 10M · fp16=20MB → Q4_K_M=5MB · 노트북·모바일 즉시 실행', theme, size=12))
    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 21. 챗봇 데모 3가지 형태
# =====================================================================

def chatbot_demo_modes(theme):
    CW, CH = 1100, 380
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '3가지 데모 형태 — CLI · Python REPL · Gradio Spaces', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '같은 GGUF 모델, 다른 껍데기 — 목적에 따라 고른다', theme))

    pal = P(theme); t = T(theme)

    modes = [
        ('input',  'CLI 한 줄',          'Ch 20 llama-cli',
         'llama-cli -m tiny-tale-q4km.gguf \\\n  -p "Once upon a time"',
         '데모·빠른 디버깅\n설치 외 설정 없음',
         '× 대화 유지 안 됨\n× 공유 불가'),
        ('model',  'Python REPL 루프',   '본 챕터 핵심',
         'from llama_cpp import Llama\nllm = Llama(model_path=...)\nwhile True: llm(input(">>> "))',
         '노트북 시연\n파라미터 실험 쉬움',
         '△ 터미널만 가능\n× URL 공유 불가'),
        ('output', 'Gradio Spaces',      '캡스톤 마지막 단계',
         'import gradio as gr\ngr.ChatInterface(fn=chat).launch()\n→ HuggingFace Spaces 배포',
         '공개 URL · 누구나 접근\nLinkedIn · 발표용',
         '○ 설정 필요\n○ 퍼블릭 노출'),
    ]

    NW, NH = 290, 180
    gap = 30
    total = len(modes) * NW + (len(modes) - 1) * gap
    left = (CW - total) // 2
    top = 86

    for i, (role, title, tag, code, pros, cons) in enumerate(modes):
        x = left + i * (NW + gap)
        # 카드
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        # 제목
        lines.append(f'  <text x="{x + NW//2}" y="{top + 26}" text-anchor="middle" font-size="15" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 42}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{tag}</text>')
        # 코드 블록 배경
        cy = top + 54
        ch_code = 52
        lines.append(f'  <rect x="{x+12}" y="{cy}" width="{NW-24}" height="{ch_code}" rx="5" fill="{t["legend_bg"]}" stroke="{t["legend_border"]}" stroke-width="0.8"/>')
        for j, cl in enumerate(code.split('\n')):
            lines.append(f'  <text x="{x+18}" y="{cy + 16 + j*16}" font-size="9.5" font-family="JetBrains Mono, monospace" fill="{t["legend_text"]}">{cl}</text>')
        # pros / cons
        py = cy + ch_code + 14
        for j, pl in enumerate(pros.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{py + j*15}" text-anchor="middle" font-size="11" fill="{pal[role]["text"]}">{pl}</text>')
        cy2 = py + len(pros.split('\n')) * 15 + 6
        for j, cl in enumerate(cons.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{cy2 + j*14}" text-anchor="middle" font-size="10" fill="{pal[role]["sub"]}">{cl}</text>')

    lines.extend(text_subtitle(CW // 2, CH - 20, '본 챕터 = Python REPL · 캡스톤 = Gradio Spaces (HF Hub 업로드 + 데모)', theme, size=12))
    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('quant-bit-tradeoff',  quant_bit_tradeoff('light'),  quant_bit_tradeoff('dark'))
    save('chatbot-demo-modes',  chatbot_demo_modes('light'),  chatbot_demo_modes('dark'))
    print('Done.')
