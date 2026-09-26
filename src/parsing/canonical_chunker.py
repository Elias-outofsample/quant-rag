from __future__ import annotations
import hashlib,re
from .models import CanonicalBlock, CanonicalChunk
from .mineru_adapter import page_roles

def _tokens(s):return len(re.findall(r"\w+|[^\w\s]",s,flags=re.UNICODE))
def _cid(*v):return 'chunk-'+hashlib.sha1('\x1f'.join(map(str,v)).encode()).hexdigest()[:16]
def _pid(*v):return 'parent-'+hashlib.sha1('\x1f'.join(map(str,v)).encode()).hexdigest()[:16]

#: Un titre numéroté en tête d'étiquette : « 6.4 THE … », « 96 Conditional … », « 3.7 THE … ».
#: Chaque composante est bornée à trois chiffres, ce qui écarte les années : « 2020 Mathematics
#: Subject Classification » ne rend pas un numéro, donc ne déclenche jamais la troncature.
_NUMERO = re.compile(r'^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s+\S')

def numero_de_titre(titre):
    """Le numéro qu'une étiquette annonce, ou ``None`` si elle n'est pas numérotée."""
    m = _NUMERO.match(str(titre or ''))
    return m.group(1) if m else None

def chemin_publie(part, chapter, section, subsection):
    """Le fil d'Ariane publié, et l'ancêtre écarté s'il contredit la section.

    Mesuré le 6 septembre 2026 (``rag/benchmark/audit_rechunk.py --chemin``) : sur les
    7 617 chunks dont ``chapter`` et ``section`` sont tous deux numérotés, **1 526 (20,0 %)**
    portent un chapitre dont le numéro contredit celui de la section — et l'échantillon montre
    que le chapitre est **fabriqué** par la regex de sommaire de ``mineru_adapter`` :
    ``96 Conditional Heteroscedastic Models`` quand il n'existe pas de chapitre 96,
    ``14 Application,`` qui est un fragment de ligne de sommaire. **La section, elle, est
    juste.** Trois manuels portent 80 % des cas.

    On ne recolle donc pas l'ancêtre — on ne saurait pas sur quoi — on **cesse de le publier**.
    Un chemin tronqué à la section dit moins ; un chemin faux dit autre chose, et il l'écrit
    dans le vecteur, dans l'en-tête servi et dans le texte lexical de BM25 à la fois.

    **Seul ``title_path`` est tronqué.** Les champs ``part``/``chapter``/``subsection`` restent
    ce que le parseur a dit — c'est de la provenance — et ``parent_id`` n'est pas touché : il
    dérive de ``chapter``, et le changer déplacerait les frontières de fusion, donc le
    découpage, bien au-delà de la variable déclarée.

    Retourne ``(title_path, chapitre_ecarte)``.
    """
    n_section, n_chapitre = numero_de_titre(section), numero_de_titre(chapter)
    ecarte = None
    if n_section and n_chapitre and n_section.split('.')[0] != n_chapitre.split('.')[0]:
        ecarte, chapter = chapter, None
    return ' > '.join(x for x in (part, chapter, section, subsection) if x), ecarte
def make_chunks(doc,blocks,target=800,hard_max=1200):
    roles=page_roles(blocks,doc.page_count); state={'part':None,'chapter':None,'section':None,'subsection':None,'chapter_key':None,'chapter_title_key':None,'document_region':'body'}; groups=[]; current=[]
    def flush():
        nonlocal current
        if current:groups.append((dict(state),list(current)));current=[]
    expanded=[]
    for b in blocks:
        if b.content_type=='table' and _tokens(b.text)>hard_max:
            rows=re.findall(r'<tr.*?</tr>',b.text,flags=re.I|re.S)
            if rows:
                prefix=b.text.split('<table',1)[0].strip(); acc=prefix; n=0
                table_parent=_pid(doc.document_id,'table',b.block_id)
                for row in rows:
                    candidate=acc+'\n'+row
                    if _tokens(candidate)>target and acc!=prefix:
                        expanded.append(CanonicalBlock(block_id=f'{b.block_id}-part{n}',document_id=b.document_id,page_idx=b.page_idx,block_type='table',text=acc,bbox=b.bbox,heading_level=None,parent_heading=b.parent_heading,content_type='table',raw_text=acc,part=b.part,chapter=b.chapter,section=b.section,subsection=b.subsection,image_refs=list(b.image_refs),metadata={**b.metadata,'table_parent_id':table_parent,'table_full_text':b.text,'table_part':n})); n+=1
                        acc=prefix+'\n'+row
                    else: acc=candidate
                if acc!=prefix: expanded.append(CanonicalBlock(block_id=f'{b.block_id}-part{n}',document_id=b.document_id,page_idx=b.page_idx,block_type='table',text=acc,bbox=b.bbox,heading_level=None,parent_heading=b.parent_heading,content_type='table',raw_text=acc,part=b.part,chapter=b.chapter,section=b.section,subsection=b.subsection,image_refs=list(b.image_refs),metadata={**b.metadata,'table_parent_id':table_parent,'table_full_text':b.text,'table_part':n}))
                continue
        if b.content_type in ('text','heading') and _tokens(b.text)>hard_max:
            words=b.text.split()
            piece_words=min(500,target)
            for n in range(0,len(words),piece_words):
                part=' '.join(words[n:n+piece_words])
                expanded.append(CanonicalBlock(block_id=f'{b.block_id}-part{n//piece_words}',document_id=b.document_id,page_idx=b.page_idx,block_type=b.block_type,text=part,bbox=b.bbox,heading_level=b.heading_level,parent_heading=b.parent_heading,content_type=b.content_type,raw_text=part,part=b.part,chapter=b.chapter,section=b.section,subsection=b.subsection,image_refs=list(b.image_refs),metadata=dict(b.metadata)))
        else: expanded.append(b)
    for b in expanded:
        if b.metadata.get('document_region') in ('references','bibliography','backmatter'):
            region=b.metadata['document_region']
            if state['document_region']!=region:
                flush(); state.update(part=None,chapter=None,section=None,subsection=None,chapter_key=None,chapter_title_key=None,document_region=region)
            if b.content_type in ('header','footer') or b.block_type in ('page_number','page_footnote'):
                continue
            current.append(b)
            if _tokens('\n'.join(x.text for x in current))>=target: flush()
            continue
        state['document_region']='body'
        if not roles.get(b.page_idx,('content',True))[1]:
            flush(); current.append(b); flush(); continue
        if b.content_type in ('header','footer') or b.block_type in ('page_number','page_footnote'):
            continue
        if b.content_type == 'table':
            flush(); current.append(b); flush(); continue
        if b.content_type in ('heading','title'):
            # `flush()` AVANT la mise à jour d'état : le tampon accumulé garde l'étiquette de
            # la section où il a été écrit. Dans l'ordre inverse — celui du dépôt jusqu'au
            # 6 septembre 2026 — le texte de la section précédente recevait l'étiquette de la
            # section qui commence. Mesuré : « le chunk ne s'ouvre pas sur sa section »
            # 57,1 % -> 10,9 %, et les classes A- et B tombent à zéro par construction, tout
            # titre ouvrant désormais son groupe (rag/benchmark/audit_rechunk.py --etiquettes).
            flush()
            if any((b.part,b.chapter,b.section,b.subsection)):
                state.update(part=b.part,chapter=b.chapter,section=b.section,subsection=b.subsection)
                state['chapter_key']=b.metadata.get('structural_identity_key',state.get('chapter_key'))
                state['chapter_title_key']=b.metadata.get('structural_title_key',state.get('chapter_title_key'))
            else:
                level=b.heading_level or 2
                if re.match(r'(?i)^(chapter|appendix)\b',b.text):state.update(chapter=b.text,section=None,subsection=None)
                elif level<=1:
                    if re.match(r'(?i)^part\b',b.text): state.update(part=b.text,chapter=None,section=None,subsection=None)
                    else: state.update(chapter=b.text,section=None,subsection=None)
                elif level==2:state['section']=b.text;state['subsection']=None
                else:state['subsection']=b.text
            current.append(b); continue
        if current and (_tokens('\n'.join(x.text for x in current))+_tokens(b.text)>target) and b.content_type not in ('table','equation'):
            flush()
        current.append(b)
        if _tokens('\n'.join(x.text for x in current))>=target:flush()
    flush(); chunks=[]
    for i,(st,bs) in enumerate(groups):
        text='\n\n'.join(b.text for b in bs if b.text).strip()
        if not text:continue
        pages=[b.page_idx for b in bs]; types={b.content_type for b in bs}; ct=next(iter(types)) if len(types)==1 else 'mixed'
        eligible=all(roles.get(p,('content',True))[1] for p in pages) and all(b.metadata.get('document_region','body') not in ('references','bibliography','backmatter','index') for b in bs)
        table_parent=next((b.metadata.get('table_parent_id') for b in bs if b.metadata.get('table_parent_id')),None)
        chapter_key=st.get('chapter_key') or st.get('chapter')
        parent=table_parent or _pid(doc.document_id,'chapter',chapter_key) if st.get('chapter') else table_parent or _pid(doc.document_id,'section',st.get('part'),st.get('section')) if st.get('section') else table_parent or _pid(doc.document_id,'part',st.get('part')) if st.get('part') else table_parent or _pid(doc.document_id,'document')
        title_path,chapitre_ecarte=chemin_publie(st.get('part'),st.get('chapter'),st.get('section'),st.get('subsection'))
        image_refs=sorted({ref for b in bs for ref in b.image_refs})
        chunks.append(CanonicalChunk(chunk_id=_cid(doc.document_id,i,text),document_id=doc.document_id,parent_id=parent,previous_chunk_id=None,next_chunk_id=None,page_start=min(pages),page_end=max(pages),part=st.get('part'),chapter=st.get('chapter'),section=st.get('section'),subsection=st.get('subsection'),text=text,title_path=title_path or None,content_type=ct,token_count=_tokens(text),rag_eligible=eligible,image_refs=image_refs,metadata={'block_ids':[b.block_id for b in bs],'page_roles':{str(p):roles[p][0] for p in sorted(set(pages))},'table_parent_id':table_parent,'structural_identity_key':chapter_key if st.get('chapter') else None,'structural_title_key':st.get('chapter_title_key') if st.get('chapter') else None,'canonical_structural_key':chapter_key if st.get('chapter') else None,'document_region':bs[0].metadata.get('document_region','body'),'chapitre_ecarte':chapitre_ecarte}))
    merged=[]
    for c in chunks:
        if merged and c.rag_eligible and merged[-1].rag_eligible and c.content_type in ('text','mixed') and merged[-1].content_type in ('text','mixed') and c.parent_id==merged[-1].parent_id and _tokens(merged[-1].text+'\n\n'+c.text)<=target:
            prev=merged[-1];prev.text+='\n\n'+c.text;prev.page_end=max(prev.page_end,c.page_end);prev.token_count=_tokens(prev.text);prev.content_type='mixed' if prev.content_type!=c.content_type else prev.content_type;prev.image_refs=sorted(set(prev.image_refs+c.image_refs));prev.metadata['block_ids']+=c.metadata.get('block_ids',[])
        else: merged.append(c)
    chunks=merged
    for i,c in enumerate(chunks):
        c.previous_chunk_id=chunks[i-1].chunk_id if i else None;c.next_chunk_id=chunks[i+1].chunk_id if i+1<len(chunks) else None
    return chunks,roles

def build_parents(doc,blocks,chunks):
    parents={}
    for c in chunks:
        pid=c.parent_id
        if pid not in parents:
            typ='table' if c.metadata.get('table_parent_id') else 'module' if c.chapter and c.chapter.lower().startswith('module ') else 'chapter' if c.chapter else 'section' if c.section else 'part' if c.part else 'document'
            identity=c.metadata.get('structural_identity_key')
            variants=doc.metadata.get('structural_variants',{}).get(identity,[]) if identity else []
            parents[pid]={'parent_id':pid,'document_id':doc.document_id,'parent_type':typ,'title':c.chapter or c.section or c.part or doc.title,'display_title':c.chapter or c.section or c.part or doc.title,'structural_identity_key':identity,'structural_title_key':c.metadata.get('structural_title_key'),'canonical_structural_key':c.metadata.get('canonical_structural_key'),'metadata':{'title_variants':variants},'title_path':c.title_path or doc.title,'page_start':c.page_start,'page_end':c.page_end,'child_chunk_ids':[],'text':''}
        p=parents[pid];p['page_start']=min(p['page_start'],c.page_start);p['page_end']=max(p['page_end'],c.page_end);p['child_chunk_ids'].append(c.chunk_id);p['text']+=(('\n\n' if p['text'] else '')+c.text)
    for b in blocks:
        pid=b.metadata.get('table_parent_id')
        if pid and pid not in parents:
            parents[pid]={'parent_id':pid,'document_id':doc.document_id,'parent_type':'table','title':b.metadata.get('caption') or 'Table','title_path':' > '.join(x for x in (b.part,b.chapter,b.section) if x),'page_start':b.page_idx,'page_end':b.page_idx,'child_chunk_ids':[],'text':b.metadata.get('table_full_text',b.text)}
    for c in chunks:
        if c.parent_id in parents and c.chunk_id not in parents[c.parent_id]['child_chunk_ids']: parents[c.parent_id]['child_chunk_ids'].append(c.chunk_id)
    for p in parents.values():
        child_pages=sorted({next(c.page_start for c in chunks if c.chunk_id==cid) for cid in p['child_chunk_ids']})
        gaps=[right-left for left,right in zip(child_pages,child_pages[1:])]
        p['parent_span_pages']=p['page_end']-p['page_start']+1
        p['max_gap_between_children']=max(gaps,default=0)
        p['parent_child_density']=len(child_pages)/p['parent_span_pages'] if p['parent_span_pages'] else 0
        p['warnings']=['suspicious_parent_gap'] if p['max_gap_between_children']>20 else []
    return list(parents.values())
