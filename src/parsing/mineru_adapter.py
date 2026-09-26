"""Convert MinerU structured JSON into canonical objects."""
from __future__ import annotations
import hashlib, html, json, re, unicodedata
from pathlib import Path
from typing import Any
from .models import CanonicalBlock, CanonicalDocument

def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()
def _id(prefix,*values): return prefix+'-'+hashlib.sha1('\x1f'.join(map(str,values)).encode()).hexdigest()[:16]
def _text(v):
    if v is None:return ''
    if isinstance(v,list):return ' '.join(_text(x) for x in v).strip()
    return html.unescape(str(v)).strip()
def _bbox(v):
    try:return [float(x) for x in v[:4]] if isinstance(v,(list,tuple)) and len(v)>=4 else None
    except (TypeError,ValueError):return None
def _entry_text(e):
    typ=e.get('type','')
    if typ=='table':
        body=_text(e.get('table_body')); cap=_text(e.get('table_caption'))
        return (('Table: '+cap+'\n') if cap else '')+body
    return _text(e.get('text') or e.get('content') or e.get('equation') or e.get('latex'))
def _ctype(typ):
    typ=str(typ).lower()
    if typ in ('header','footer','page_header','page_footer'):return typ
    if typ in ('title','table','equation','image'):return typ
    if typ=='chart':return 'mixed'
    return 'text'
def _heading_level(entry, text):
    raw=entry.get('text_level')
    try: level=int(raw) if raw is not None else None
    except (TypeError,ValueError): level=None
    if level==1 and entry.get('page_idx',0)<3 and len(text)<80: return None
    return level
def _norm_heading(text):
    return re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()
def structural_title_key(title):
    text=unicodedata.normalize('NFKC',_text(title)).replace('’',"'").replace('‘',"'")
    text=re.sub(r'[^\w]+',' ',text.casefold(),flags=re.UNICODE)
    return re.sub(r'\s+',' ',text).strip()
def structural_identity_key(kind, title, part=None):
    """Identity favours explicit chapter or module numbering over OCR title variants."""
    text=structural_title_key(title)
    number=re.match(r'(?:(?:chapter)\s+)?(\d{1,2})\b',text)
    module=re.match(r'module\s+([a-g])\b',text)
    identity=number.group(1) if number else module.group(1) if module else text
    context=_norm_heading(part or '')
    return ':'.join(x for x in (kind,context,identity) if x)
def canonical_structural_key(kind, title, part=None):
    return structural_identity_key(kind,title,part)+':'+structural_title_key(title)
def _content_list(out):
    files=list(Path(out).rglob('*_content_list.json'))
    if not files:raise FileNotFoundError('No MinerU content_list.json under '+str(out))
    data=json.loads(files[0].read_text(encoding='utf-8'))
    if not isinstance(data,list):raise ValueError('Unexpected content list shape')
    return [x for x in data if isinstance(x,dict)]
def _middle_meta(out):
    files=list(Path(out).rglob('*_middle.json'))
    if not files:return {}
    try:
        d=json.loads(files[0].read_text(encoding='utf-8')); return {'middle_pages':len(d.get('pdf_info',[])),'middle_version':d.get('_version_name')}
    except (OSError,json.JSONDecodeError):return {}
def parse_mineru_json(pdf_path, output_dir):
    path,out=Path(pdf_path),Path(output_dir); digest=_sha256(path)
    did=_id('doc',digest,'mineru','3.4.5','canonical-v1'); blocks=[]; stack=[]
    for idx,e in enumerate(_content_list(out)):
        typ=str(e.get('type','text')); text=_entry_text(e)
        if not text and typ not in ('image','chart'):continue
        level=_heading_level(e,text); ct=_ctype(typ)
        if level is not None and typ not in ('header','footer'): ct='heading'
        if ct=='heading':
            level=level or (1 if re.match(r'(?i)^(part|chapter|appendix|preface|references|index)\b',text) else 2)
            stack=stack[:max(0,level-1)]; stack.append(text)
        page=int(e.get('page_idx',0)); meta={k:v for k,v in e.items() if k not in ('text','bbox','page_idx','type')}
        meta['raw_mineru_type']=typ
        if e.get('text_level') is not None: meta['text_level']=e.get('text_level')
        if typ=='table':meta.update(table_html=e.get('table_body'),caption=e.get('table_caption'))
        if typ=='equation':meta['equation_text']=text
        image_refs=[]
        if e.get('img_path'): image_refs=[str(e['img_path']).replace('\\','/')]
        blocks.append(CanonicalBlock(block_id=_id('block',did,idx,page,typ,text), document_id=did, page_idx=page,
            block_type=typ, text=text, bbox=_bbox(e.get('bbox')), heading_level=level,
            parent_heading=stack[-2] if len(stack)>1 else None, content_type=ct,
            raw_text=text, image_refs=image_refs, metadata=meta))
    chapter_catalog={}
    for b in blocks:
        for m in re.finditer(r'\bCHAPTER\s+(\d{1,2})\s+([A-Z][A-Z0-9 :;,\-()]+)',b.text,re.I):
            candidate=re.sub(r'\s+',' ',m.group(2)).strip()
            if candidate==candidate.upper(): chapter_catalog[int(m.group(1))]=candidate.title()
    existing_numbers={int(m.group(1)) for b in blocks if b.content_type=='heading' for m in [re.match(r'^(\d{1,2})\s+',b.text)] if m}
    synthetic=[]
    for b in blocks:
        m=re.fullmatch(r'Chapter\s+(\d{1,2})',b.text.strip(),re.I)
        if not m: continue
        number=int(m.group(1)); title_text=chapter_catalog.get(number)
        if number in existing_numbers or not title_text: continue
        synthetic.append(CanonicalBlock(block_id=_id('block',did,'chapter-header',number,b.page_idx,title_text),document_id=did,page_idx=b.page_idx,block_type='inferred_heading',text=f'{number} {title_text}',bbox=b.bbox,heading_level=1,parent_heading=None,content_type='heading',raw_text=title_text,metadata={'heading_source':'inferred','heading_confidence':8,'heading_reasons':['body chapter header','catalog title','number sequence']}))
    blocks.extend(synthetic)
    blocks.sort(key=lambda b:(b.page_idx, b.bbox[1] if b.bbox else 10**9, 0 if b.block_type=='inferred_heading' else 1, b.block_id))
    first=[b.text for b in blocks[:60] if b.text]; title=None; subtitle=None; authors=[]
    for t in first:
        low=t.lower()
        if title is None and any(x in low for x in ('finding alphas','dynamic hedging','market microstructure')):title=t
        if 'a quantitative approach to building trading strategies' in low:subtitle='A Quantitative Approach to Building Trading Strategies'
        if 'igor tulchinsky' in low:authors=['Igor Tulchinsky et al.']
    stem=path.stem.lower()
    if 'finding-alphas' in stem:
        title='Finding Alphas'; subtitle='A Quantitative Approach to Building Trading Strategies'; authors=['Igor Tulchinsky et al.']
    elif 'dynamic hedging' in stem:
        title='Dynamic Hedging'; subtitle='Managing Vanilla and Exotic Options'; authors=['Nassim Taleb']
    elif 'market-microstructure' in stem:
        title='Market Microstructure in Practice'; authors=['Charles-Albert Lehalle','Sophie Laruelle']
    title=title or re.sub(r'[_-]+',' ',path.stem).strip()
    provisional_pages=max((b.page_idx for b in blocks),default=-1)+1
    roles=page_roles(blocks,provisional_pages)
    # Dynamic Hedging's notes reproduce chapter and module names.  Identify this
    # terminal notes section from its content and its later Bibliography, rather
    # than from a page number.
    bibliography_pages=[b.page_idx for b in blocks if b.page_idx>provisional_pages*.7 and b.text.strip().casefold()=='bibliography']
    late_prefaces=[b.page_idx for b in blocks if b.page_idx>provisional_pages*.7 and b.text.strip().casefold()=='preface']
    backmatter_start=None
    if bibliography_pages:
        bibliography_start=min(bibliography_pages)
        candidates=[p for p in late_prefaces if p<bibliography_start]
        if candidates:
            candidate=max(candidates)
            nearby=' '.join(b.text for b in blocks if candidate<=b.page_idx<=min(candidate+12,bibliography_start)).casefold()
            if 'chapter 1' in nearby and 'chapter' in nearby:
                backmatter_start=candidate
    regions={}
    for p in range(provisional_pages):
        role=roles.get(p,('content',True))[0]
        if role=='index': region='index'
        elif bibliography_pages and p>=min(bibliography_pages): region='bibliography'
        elif backmatter_start is not None and p>=backmatter_start: region='backmatter'
        elif role in ('cover','copyright','dedication','toc','blank'): region='frontmatter' if role!='toc' else 'toc'
        else: region='body'
        regions[p]=region
    for b in blocks: b.metadata['document_region']=regions[b.page_idx]
    module_on_page={}
    for b in blocks:
        m=re.fullmatch(r'Module\s+([A-G])',b.text.strip(),re.I)
        if m and b.block_type=='header': module_on_page[b.page_idx]='Module '+m.group(1).upper()
    heading_pages={}
    for b in blocks:
        if b.content_type=='heading': heading_pages.setdefault(b.text.strip().lower(),set()).add(b.page_idx)
    for b in blocks:
        if b.content_type=='heading' and len(heading_pages.get(b.text.strip().lower(),()))>=3:
            b.content_type='header'; b.metadata['heading_source']='running_header'; b.metadata['semantic_heading']=False
    toc_hints=[]; toc_part=None; part_ranges=[]
    front_text=' '.join(b.text for b in blocks if b.page_idx<20)
    for m in re.finditer(r'Part\s+([IVXLC]+).*?Chapters\s+(\d+)\s*[-–]\s*(\d+)',front_text,re.I):
        part_ranges.append((int(m.group(2)),int(m.group(3)),'PART '+m.group(1).upper()))
    for b in sorted(blocks,key=lambda x:(x.page_idx,x.block_id)):
        if roles.get(b.page_idx,('content',True))[0]!='toc': continue
        text=b.text
        part_match=re.search(r'PART\s+([IVXLC]+)(?:\s+[^\n]*)?',text,re.I)
        if part_match: toc_part='PART '+part_match.group(1).upper()
        for m in re.finditer(r'\b(\d{1,2})\s+([A-Z][A-Za-z][^\n]{3,90}?)(?=\s+\d{1,3}\b|$)',text):
            toc_title=re.sub(r'\s+',' ',m.group(2)).strip(' .')
            toc_hints.append((_norm_heading(toc_title),f'{m.group(1)} {toc_title}',toc_part))
    hierarchy={'part':None,'chapter':None,'section':None,'subsection':None,'chapter_key':None}; seen_module_pages=set()
    closed_nodes={}; reopen_attempts=[]; structural_variants={}
    def set_chapter(display, kind='chapter', confidence=7):
        key=structural_identity_key(kind,display,hierarchy['part'])
        title_key=structural_title_key(display)
        previous=hierarchy.get('chapter_key')
        structural_variants.setdefault(key,set()).add(display)
        if previous==key:
            return True
        if previous and previous!=key: closed_nodes[previous]=current_page
        if key in closed_nodes and key!=previous:
            reopen_attempts.append({'page':current_page+1,'candidate':display,'canonical_key':key,'closed_at_page':closed_nodes[key]+1,'action':'ignored','reason':'closed structural node'})
            return False
        hierarchy.update(chapter=display,chapter_key=key,section=None,subsection=None)
        return True
    for b in blocks:
        if b.content_type not in ('heading','title'): continue
        current_page=b.page_idx
        if roles.get(b.page_idx,('content',True))[0]=='toc':
            b.metadata['heading_source']='toc'; continue
        region=regions[b.page_idx]
        if region in ('references','bibliography','backmatter','index'):
            b.metadata.update(heading_source='backmatter_reference',heading_confidence=0,semantic_heading=False)
            candidate=re.sub(r'\s+',' ',b.text).strip()
            candidate_key=structural_identity_key('chapter',candidate,hierarchy['part'])
            if candidate_key in closed_nodes or re.match(r'(?i)^chapter\s+\d+|^\d+\s+',candidate):
                reopen_attempts.append({'page':b.page_idx+1,'candidate':candidate,'canonical_key':candidate_key,'closed_at_page':closed_nodes.get(candidate_key),'action':'ignored','reason':region})
            continue
        b.metadata['heading_source']='body'
        s=re.sub(r'\s+',' ',b.text).strip(); up=s.upper()
        module_label=module_on_page.get(b.page_idx)
        if module_label and b.page_idx not in seen_module_pages and b.heading_level in (1,2) and not s.isupper():
            hierarchy['part']='PART IV'; set_chapter(f'{module_label} {s}','module',8)
            seen_module_pages.add(b.page_idx)
            b.metadata['structural_kind']='module'; b.metadata['structural_identity_key']=hierarchy['chapter_key']; b.metadata['structural_title_key']=structural_title_key(hierarchy['chapter']); b.metadata['canonical_structural_key']=canonical_structural_key('module',hierarchy['chapter'],hierarchy['part']); b.metadata['heading_confidence']=8; b.metadata['heading_reasons']=['body module header','text_level heading']
            b.part,b.chapter,b.section,b.subsection=(hierarchy['part'],hierarchy['chapter'],hierarchy['section'],hierarchy['subsection'])
            b.parent_heading=hierarchy['chapter']; continue
        explicit_number=re.match(r'^(\d{1,2})\s+',s)
        if explicit_number:
            hint=next((h for h in toc_hints if h[1].startswith(explicit_number.group(1)+' ')),None)
        else:
            hint=next((h for h in toc_hints if h[0]==_norm_heading(s) or h[0] in _norm_heading(s) or _norm_heading(s) in h[0]),None)
        if b.block_type=='inferred_heading':
            number=int(re.match(r'^(\d+)',s).group(1)); range_part=next((part for lo,hi,part in part_ranges if lo<=number<=hi),hierarchy['part'])
            hierarchy['part']=range_part; set_chapter(s,'chapter',8)
        elif re.match(r'^PART\s+[IVXLC]+',up): hierarchy.update(part=s,chapter=None,section=None,subsection=None)
        elif hint and (b.heading_level==1 or re.match(r'^\d+\s+',s)):
            number=int(re.match(r'^(\d+)',hint[1]).group(1)) if re.match(r'^(\d+)',hint[1]) else None
            range_part=next((part for lo,hi,part in part_ranges if number is not None and lo<=number<=hi),None)
            hierarchy['part']=range_part or hint[2] or hierarchy['part']; set_chapter(hint[1],'chapter',7)
        elif hint and hierarchy['chapter']:
            hierarchy.update(section=s,subsection=None)
        elif b.heading_level==1 and re.match(r'^MODULE(?:\s+[A-G])?',s,re.I): hierarchy['part']='PART IV'; set_chapter(s,'module',7)
        elif re.match(r'^(CHAPTER\s+\d+|\d+\s+[A-Z])',s,re.I): set_chapter(s,'chapter',7)
        elif b.page_idx<3 and b.content_type=='title': continue
        elif hierarchy['chapter'] and (b.heading_level or 2)<=2: hierarchy.update(section=s,subsection=None)
        elif hierarchy['chapter']: hierarchy['subsection']=s
        b.part,b.chapter,b.section,b.subsection=(hierarchy['part'],hierarchy['chapter'],hierarchy['section'],hierarchy['subsection'])
        if hierarchy.get('chapter_key'):
            b.metadata['structural_identity_key']=hierarchy['chapter_key']
            b.metadata['structural_title_key']=structural_title_key(hierarchy['chapter'])
            b.metadata['canonical_structural_key']=canonical_structural_key('chapter',hierarchy['chapter'],hierarchy['part'])
        b.parent_heading=hierarchy['chapter'] or hierarchy['part']
    pages=max((b.page_idx for b in blocks),default=-1)+1
    meta=_middle_meta(out)
    meta['document_regions']={str(page+1):region for page,region in regions.items()}
    meta['structural_reopen_attempts']=reopen_attempts
    meta['structural_variants']={key:sorted(values) for key,values in structural_variants.items()}
    try:
        import fitz; meta['pdf_metadata']={k:v for k,v in fitz.open(path).metadata.items() if v}
    except Exception:pass
    if 'finding-alphas' in stem:meta.update(edition='Second Edition',publisher='Wiley',publication_year=2020,first_edition_year=2015)
    elif 'dynamic hedging' in stem:meta.update(publisher='John Wiley & Sons')
    elif 'market-microstructure' in stem:meta.update(edition='Second Edition',publisher='World Scientific',publication_year=2018)
    edition=meta.get('edition'); publisher=meta.get('publisher'); publication_year=meta.get('publication_year')
    doc=CanonicalDocument(document_id=did,source_path=str(path),filename=path.name,sha256=digest,
        title=title,subtitle=subtitle,authors=authors,publication_year=publication_year,page_count=pages,
        parser_backend='mineru',parser_version='3.4.5',edition=edition,publisher=publisher,metadata=meta)
    return doc,blocks
def page_roles(blocks,page_count):
    roles={}
    toc_candidates=[]; toc_markers=[]
    for p in range(page_count):
        t=' '.join(b.text for b in blocks if b.page_idx==p).lower(); role='content'; ok=True
        if not t:role,ok='blank',False
        elif p==0:role,ok='cover',False
        elif 'table of contents' in t or (re.search(r'\bcontents\b',t) and len(re.findall(r'\.\s*\d+',t))>=3):role,ok='toc',False
        elif 'copyright' in t or 'all rights reserved' in t:role,ok='copyright',False
        elif 'dedication' in t and p < 10:role,ok='dedication',False
        roles[p]=(role,ok)
        numeric_refs=len(re.findall(r'\b\d{1,3}\b',t))
        if 3 <= p < 30 and numeric_refs >= 10: toc_candidates.append(p)
        if 3 <= p < 30 and re.search(r'\bcontents\b',t): toc_markers.append(p)
    if len(toc_candidates)>=2:
        lo=min(toc_markers) if toc_markers else min(toc_candidates)
        marker_hi=max(toc_markers) if toc_markers else None
        toc_end=[p for p in toc_candidates if p>=lo and (marker_hi is None or p<=marker_hi)]
        hi=max(toc_end, default=None)
        if hi is not None and hi-lo<=12:
            for p in range(lo,hi+1): roles[p]=('toc',False)
    index_start=next((b.page_idx for b in blocks if b.page_idx>page_count*.8 and b.text.strip().lower()=='index' and b.block_type in ('header','inferred_heading','text')),None)
    if index_start is not None:
        for p in range(index_start,page_count):
            t=' '.join(b.text for b in blocks if b.page_idx==p).lower()
            short_refs=len(re.findall(r'\b[a-z][a-z -]{2,35},?\s+\d{1,3}\b',t))
            if p==index_start or short_refs>=5 or any(b.block_type=='header' and b.text.strip().lower()=='index' for b in blocks if b.page_idx==p): roles[p]=('index',False)
    return roles
