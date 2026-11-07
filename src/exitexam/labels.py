"""English labels for charts (Pashto text needs shaping libraries; English always renders correctly)."""
from .cleaning import norm_text

PROVINCE_EN = {norm_text(k): v for k, v in {
    'کابل': 'Kabul', 'بلخ': 'Balkh', 'هرات': 'Herat', 'کندهار': 'Kandahar', 'ننګرهار': 'Nangarhar',
    'هلمند': 'Helmand', 'خوست': 'Khost', 'غزنی': 'Ghazni', 'پروان': 'Parwan', 'بغلان': 'Baghlan',
    'جوزجان': 'Jawzjan', 'کاپیسا': 'Kapisa', 'پکتیا': 'Paktia', 'کندز': 'Kunduz', 'abroad': 'Abroad', 'unknown': 'Unknown'}.items()}

COUNTRY_EN = {norm_text(k): v for k, v in {
    'پاکستان': 'Pakistan', 'چین': 'China', 'روسیه': 'Russia', 'قزاقستان': 'Kazakhstan',
    'تاجیکستان': 'Tajikistan', 'ترکیه': 'Turkey', 'قرغزستان': 'Kyrgyzstan', 'سودان': 'Sudan',
    'هندوستان': 'India', 'مصر': 'Egypt', 'ایران': 'Iran', 'unknown': 'Other / unknown'}.items()}

_INST = {
    # public (state) institutes: the key is the text used in the files
    'طبی کابل ابو علی ابن سینا': 'Kabul Medical Univ. (Avicenna)', 'شیخ زاید خوست': 'Khost (Sheikh Zayed)',
    'البیرونی': 'Al-Biruni', 'پکتیا': 'Paktia', 'بلخ': 'Balkh', 'هرات': 'Herat', 'کندهار': 'Kandahar',
    'ننګرهار': 'Nangarhar', 'کندز': 'Kunduz',
    # private institutes
    'روښان': 'Rowshan', 'اریانا': 'Ariana', 'سپین غر': 'Spin Ghar', 'تاج': 'Taj', 'افغان پامیر': 'Afghan Pamir',
    'نعمان سادات': 'Noman Sadat', 'افغان سویس': 'Afghan Swiss', 'شفا': 'Shifa', 'بایزید روښان': 'Bayazid Rowshan',
    'رهنورد': 'Rahnaward', 'وادی هلمند': 'Wadi Helmand', 'مولانا جلال الدین محمد بلخی': 'Mawlana Balkhi',
    'خاتم النبیین': 'Khatam al-Nabiyyin', 'بست': 'Bost', 'احمد شاه ابدالی': 'Ahmad Shah Abdali',
    'میرویس نیکه': 'Mirwais Nika', 'ملی': 'Melli', 'اریا': 'Aria', 'هریوا': 'Hariwa', 'حکیم سنایی': 'Hakim Sanai',
    'میهن': 'Mihan', 'مسلم': 'Muslim', 'دانش': 'Danish', 'غالب': 'Ghalib', 'کاتب': 'Katib', 'ملالی': 'Malalai',
    'زاول': 'Zawul', 'شهید سیدجان': 'Shahid Sayyed Jan', 'شیرزی': 'Shirzai', 'چراغ': 'Cheragh',
    'ارکوزیا': 'Arkozia', 'جامی': 'Jami', 'طلوع سعادت': 'Tolu-e-Saadat',
}
INSTITUTE_EN = {norm_text(k): v for k, v in _INST.items()}
TYPE_EN = {'public': 'Public', 'private': 'Private', 'abroad': 'Abroad', 'unknown': 'Unknown'}
FIELD_EN = {'medicine': 'Medicine', 'stomatology': 'Stomatology', 'unknown': 'Unknown'}


def institute_label(inst_type, province, institute):
    inst = INSTITUTE_EN.get(institute, 'Inst. ?')
    if inst_type == 'private':
        return f'{PROVINCE_EN.get(province, province)} - {inst}'
    if inst_type == 'public':
        return f'{inst} (public)'
    return 'Abroad'

TERM_EN = {
    'type[Private]': 'Private vs public institute', 'type[Abroad]': 'Abroad vs public institute',
    'field_c[Stomatology]': 'Stomatology vs medicine', 'field[stomatology]': 'Stomatology vs medicine',
    'round[2023]': 'Round 2023 vs 2025', 'round[2024]': 'Round 2024 vs 2025',
    'round[2026-R1]': 'Round 2026 R1 vs 2025', 'round[2026-R2]': 'Round 2026 R2 vs 2025',
    'prior_fails_cat[1]': '1 previous failure vs none', 'prior_fails_cat[2]': '2 previous failures vs none',
    'prior_fails_cat[3+]': '3+ previous failures vs none', 'prior_fails_cat[2+]': '2+ previous failures vs none',
    'gap10': 'Per 10 points closer to pass mark (last attempt)', 'gap10_sq': '(same, squared term)',
    'rounds_since_last[2+]': 'Skipped at least one round vs came straight back',
}
