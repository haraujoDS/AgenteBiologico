import os
import re
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ==============================================================================
# CONFIGURAÇÃO DE SEGURANÇA E CHAVE (LOCAL OU NUVEM)
# ==============================================================================
# No Streamlit Cloud, a chave vem de st.secrets. Localmente, pode vir do .env
chave_gemini = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    chave_gemini = st.secrets["GEMINI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    chave_gemini = os.getenv("GEMINI_API_KEY")

if chave_gemini:
    genai.configure(api_key=chave_gemini)
else:
    st.error("Chave GEMINI_API_KEY não configurada!")

# ==============================================================================
# PROCESSAMENTO DE PDFS E MOTOR RAG LOCAL
# ==============================================================================
class ProcessadorDocumentosBio:
    def __init__(self, pasta_docs: Path):
        self.pasta_docs = pasta_docs

    def _limpar_texto(self, texto: str) -> str:
        texto = re.sub(r'(\w+)-\n(\w+)', r'\1\2', texto)
        texto = re.sub(r'\n+', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()

    def processar_ficheiros(self, tamanho_chunk: int = 700, sobreposicao: int = 100) -> List[Dict[str, Any]]:
        fragmentos_totais = []
        termos_ignorar = ["agradecimentos", "sumário", "lista de figuras", "lista de tabelas"]

        if not self.pasta_docs.exists():
            return []

        for ficheiro_pdf in self.pasta_docs.glob("*.pdf"):
            try:
                leitor = PdfReader(str(ficheiro_pdf))
                for idx_pag, pagina in enumerate(leitor.pages):
                    texto_extraido = pagina.extract_text()
                    if not texto_extraido:
                        continue

                    if idx_pag < 15 and any(t in texto_extraido.lower() for t in termos_ignorar):
                        continue

                    if "\nReferências" in texto_extraido or "\nReferences" in texto_extraido:
                        break

                    texto_limpo = self._limpar_texto(texto_extraido)
                    if len(texto_limpo) < 50:
                        continue

                    for i in range(0, len(texto_limpo), tamanho_chunk - sobreposicao):
                        chunk = texto_limpo[i:i + tamanho_chunk]
                        if len(chunk) > 100:
                            fragmentos_totais.append({
                                "fonte": f"{ficheiro_pdf.name} (Pág. {idx_pag + 1})",
                                "texto": chunk
                            })
            except Exception as e:
                print(f"[Erro PDF]: {e}")

        return fragmentos_totais


class MotorRAG:
    def __init__(self, pasta_docs: Path):
        self.processador = ProcessadorDocumentosBio(pasta_docs)
        self.fragmentos: List[Dict[str, Any]] = []
        self.vetorizador: Optional[TfidfVectorizer] = None
        self.matriz_tfidf = None
        self._indexar()

    def _indexar(self):
        self.fragmentos = self.processador.processar_ficheiros()
        if not self.fragmentos:
            return

        corpus = [f["texto"] for f in self.fragmentos]
        self.vetorizador = TfidfVectorizer(stop_words=None, lowercase=True)
        self.matriz_tfidf = self.vetorizador.fit_transform(corpus)

    def buscar(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if self.matriz_tfidf is None or self.vetorizador is None:
            return []

        query_vec = self.vetorizador.transform([query])
        similaridades = cosine_similarity(query_vec, self.matriz_tfidf).flatten()

        indices_ordenados = np.argsort(similaridades)[::-1]
        resultados = []

        for idx in indices_ordenados[:top_k]:
            score = float(similaridades[idx])
            if score >= 0.08:
                resultados.append({
                    "texto": self.fragmentos[idx]["texto"],
                    "fonte": self.fragmentos[idx]["fonte"],
                    "score": round(score, 4)
                })

        return resultados

# Cache para carregar os PDFs na memória uma única vez ao iniciar a aplicação
@st.cache_resource(show_spinner="Carregando e indexando documentos do laboratório...")
def carregar_motor_rag():
    pasta_docs = Path(__file__).resolve().parent / "documentos"
    return MotorRAG(pasta_docs)

# ==============================================================================
# INTEGRAÇÃO PUBMED & SÍNTESE COM GEMINI
# ==============================================================================
def consultar_pubmed_api(termo: str, max_resultados: int = 2) -> List[Dict[str, str]]:
    url_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    termo_codificado = urllib.parse.quote(termo)
    url_esearch = f"{url_base}esearch.fcgi?db=pubmed&term={termo_codificado}&retmode=json&retmax={max_resultados}"

    try:
        req = urllib.request.Request(url_esearch, headers={"User-Agent": "Streamlit-BioAgent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
            id_list = dados.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        id_str = ",".join(id_list)
        url_esummary = f"{url_base}esummary.fcgi?db=pubmed&id={id_str}&retmode=json"
        req_resumo = urllib.request.Request(url_esummary, headers={"User-Agent": "Streamlit-BioAgent/1.0"})

        with urllib.request.urlopen(req_resumo, timeout=5) as resposta_resumo:
            dados_resumo = json.loads(resposta_resumo.read().decode("utf-8"))
            resultados = []
            for pmid in id_list:
                item = dados_resumo.get("result", {}).get(pmid, {})
                resultados.append({
                    "fonte": f"PubMed PMID: {pmid}",
                    "titulo": item.get("title", "Título indisponível")
                })
            return resultados
    except Exception:
        return []


def gerar_resposta_sintese(pergunta: str, contexto: str, origem: str) -> str:
    prompt_completo = (
        "Você é um cientista e bioinformata sênior especializado em engenharia genética de "
        "Komagataella phaffii (Pichia pastoris), desenho de vetores e biologia sintética.\n\n"
        "Diretrizes:\n"
        "1. Responda estritamente com base no contexto técnico fornecido abaixo.\n"
        "2. Seja claro, direto e cientificamente rigoroso (destaque linhagens, genes, vetores e fenótipos mencionados).\n"
        "3. Não invente metodologias ou dados ausentes no contexto.\n\n"
        f"Origem das evidências: {origem}\n\n"
        f"Contexto recuperado:\n\"\"\"\n{contexto}\n\"\"\"\n\n"
        f"Pergunta do usuário: {pergunta}"
    )

    try:
        modelo = genai.GenerativeModel(
            model_name="gemini-3.8-flash",
            generation_config={"temperature": 0.2, "max_output_tokens": 2048}
        )
        resposta = modelo.generate_content(prompt_completo, request_options={"timeout": 25})
        return resposta.text.strip()
    except Exception as e:
        return f"Falha na comunicação com a API do Gemini. Detalhes: {str(e)}"


def agente_decisor(pergunta: str, rag: MotorRAG) -> Dict[str, Any]:
    trechos_locais = rag.buscar(pergunta, top_k=2)

    if trechos_locais:
        fontes = [t["fonte"] for t in trechos_locais]
        maior_score = trechos_locais[0]["score"]
        contexto_combinado = "\n\n---\n\n".join(
            [f"[Fonte: {t['fonte']}]: {t['texto']}" for t in trechos_locais]
        )
        resposta_final = gerar_resposta_sintese(
            pergunta, contexto_combinado, "Documentos Técnicos Locais (Teses e Artigos)"
        )
        return {
            "origem": "RAG Local (Teses & Artigos)",
            "fontes": list(set(fontes)),
            "score": maior_score,
            "resposta": resposta_final
        }

    artigos_externos = consultar_pubmed_api(pergunta, max_resultados=2)
    if artigos_externos:
        fontes = [a["fonte"] for a in artigos_externos]
        contexto_externo = "\n".join([f"- {a['titulo']} ({a['fonte']})" for a in artigos_externos])
        resposta_final = gerar_resposta_sintese(
            pergunta, contexto_externo, "API Externa do PubMed / NCBI"
        )
        return {
            "origem": "API Externa (PubMed / NCBI)",
            "fontes": fontes,
            "score": 0.50,
            "resposta": resposta_final
        }

    return {
        "origem": "Sem Evidências",
        "fontes": [],
        "score": 0.0,
        "resposta": "Não foram localizadas informações suficientes sobre essa consulta nos PDFs locais nem no PubMed."
    }

# ==============================================================================
# INTERFACE DO STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Agente Komagataella phaffii", page_icon="🧬", layout="centered")

st.title("🧬 Agente de Engenharia Genética")
st.caption("Especializado em vetores, promotores e linhagens de *Komagataella phaffii*")

rag_service = carregar_motor_rag()

if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fontes" in msg and msg["fontes"]:
            with st.expander(f"📚 Evidências ({msg.get('origem', '')})"):
                for f in msg["fontes"]:
                    st.write(f"- {f}")

if prompt := st.chat_input("Digite sua dúvida técnica..."):
    st.session_state.historico.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando acervo do laboratório e literatura..."):
            resultado = agente_decisor(prompt, rag_service)
            texto = resultado["resposta"]
            fontes = resultado["fontes"]
            origem = resultado["origem"]

            st.markdown(texto)
            if fontes:
                with st.expander(f"📚 Evidências ({origem})"):
                    for f in fontes:
                        st.write(f"- {f}")

            st.session_state.historico.append({
                "role": "assistant",
                "content": texto,
                "fontes": fontes,
                "origem": origem
            })