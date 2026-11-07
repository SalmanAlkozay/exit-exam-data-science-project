"""Builds everything for a plain-language report: python make_public_report.py  (about 1-2 minutes).
Writes outputs/public_report/:
  charts/            Pashto charts (right-to-left)
  charts_english/    the same 19 charts in English
  report_tables_pashto.docx / .xlsx   tables
  report_pashto_full.docx   the full Pashto report draft with Pashto charts and tables
  report_pashto_english_charts.docx   the same report with the ENGLISH charts + the paper's technical figures as an appendix
  charts_index_pashto.docx, charts_index_english.md   titles, sub-titles and notes (use as captions)"""
import os, re, sys
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from exitexam import public_report as PR, public_tables as PT, public_charts as PC, public_report_doc as RD


def english_index(H, path):
    lines = ['# Chart index (English)', '', 'File name, title, sub-title and note of each chart. Use them as captions.', '']
    for i, (key, html) in enumerate(H.items(), 1):
        g = lambda pat: (re.sub('<.*?>', '', re.search(pat, html, re.S).group(1)).strip() if re.search(pat, html, re.S) else '')
        lines += [f'## {i}. {PC.file_name(key)}.png', '', f'**{g(r"<h1>(.*?)</h1>")}**', '']
        if g(r'<p class="sub">(.*?)</p>'):
            lines += [g(r'<p class="sub">(.*?)</p>'), '']
        if g(r'<div class=note>(.*?)</div>'):
            lines += ['Note: ' + g(r'<div class=note>(.*?)</div>'), '']
    open(path, 'w', encoding='utf-8').write('\n'.join(lines))


if __name__ == '__main__':
    D = PR.collect()
    Hps = PC.build_charts(D, lang='ps'); n1 = len(PC.render(Hps, out_dir=PC.CHART_DIR))
    Hen = PC.build_charts(D, lang='en'); n2 = len(PC.render(Hen, out_dir=PC.CHART_DIR_EN))
    tabs = PT.build_tables(D)
    PT.write_word(tabs, os.path.join(PC.PUBLIC_DIR, 'report_tables_pashto.docx'))
    PT.write_excel(tabs, os.path.join(PC.PUBLIC_DIR, 'report_tables_pashto.xlsx'))
    PT.write_chart_index(Hps, os.path.join(PC.PUBLIC_DIR, 'charts_index_pashto.docx'))
    RD.build_report(D, os.path.join(PC.PUBLIC_DIR, 'report_pashto_full.docx'))
    RD.build_report(D, os.path.join(PC.PUBLIC_DIR, 'report_pashto_english_charts.docx'), chart_lang='en', technical=True)
    english_index(Hen, os.path.join(PC.PUBLIC_DIR, 'charts_index_english.md'))
    print(n1, 'Pashto charts,', n2, 'English charts,', len(tabs), 'tables ->', PC.PUBLIC_DIR)
