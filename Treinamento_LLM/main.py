import os
import re
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai

# ==============================================================================
# CONFIGURAÇÃO DE AMBIENTE & LLM
# ==============================================================================
load_dotenv()
chave_gemini = os.getenv("GEMINI_API_KEY")

if chave_gemini:
    genai.configure(api_key=chave_gemini)
else:
    print("[Aviso] GEMINI_API_KEY não foi encontrada no ficheiro .env!")


# ==============================================================================
# BLOCO 1: CONTRATOS DE DADOS DA API (PYDANTIC)
# ==============================================================================
class RequisicaoConsulta(BaseModel):
    pergunta: str
    usuario_id: int


class RespostaConsulta(BaseModel):
    usuario_id: int
    pergunta: str
    origem_dados: str
    fontes_consultadas: List[str]
    resposta: str
    score_confianca: float


# ==============================================================================
# BLOCO 2: PROCESSAMENTO E LIMPEZA DOS DOCUMENTOS PDF
# ==============================================================================
class ProcessadorDocumentosBio:
    def __init__(self, pasta_docs: Path):
        self.pasta_docs = pasta_docs

    def _limpar_texto(self, texto: str) -> str:
        """Remove quebras de linha forçadas e une palavras cortadas por hífen."""
        texto = re.sub(r'(\w+)-\n(\w+)', r'\1\2', texto)
        texto = re.sub(r'\n+', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()

    def processar_ficheiros(self, tamanho_chunk: int = 700, sobreposicao: int = 100) -> List[Dict[str, Any]]:
        """Extrai, limpa e fragmenta o texto dos ficheiros PDF da pasta."""
        fragmentos_totais = []
        termos_ignorar = ["agradecimentos", "sumário", "lista de figuras", "lista de tabelas"]

        for ficheiro_pdf in self.pasta_docs.glob("*.pdf"):
            try:
                leitor = PdfReader(str(ficheiro_pdf))
                for idx_pag, pagina in enumerate(leitor.pages):
                    texto_extraido = pagina.extract_text()
                    if not texto_extraido:
                        continue

                    # Ignora páginas iniciais administrativas/preliminares
                    if idx_pag < 15 and any(t in texto_extraido.lower() for t in termos_ignorar):
                        continue

                    # Interrompe se atingir a lista de referências bibliográficas
                    if "\nReferências" in texto_extraido or "\nReferences" in texto_extraido:
                        break

                    texto_limpo = self._limpar_texto(texto_extraido)
                    if len(texto_limpo) < 50:
                        continue

                    # Fragmentação por janela deslizante
                    for i in range(0, len(texto_limpo), tamanho_chunk - sobreposicao):
                        chunk = texto_limpo[i:i + tamanho_chunk]
                        if len(chunk) > 100:
                            fragmentos_totais.append({
                                "fonte": f"{ficheiro_pdf.name} (Pág. {idx_pag + 1})",
                                "texto": chunk
                            })
            except Exception as e:
                print(f"[Erro] Falha ao ler {ficheiro_pdf.name}: {e}")

        return fragmentos_totais


# ==============================================================================
# BLOCO 3: MECANISMO DE BUSCA VETORIAL LOCAL (RAG)
# ==============================================================================
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
            print("[RAG] Nenhum documento válido encontrado para indexação.")
            return

        corpus = [f["texto"] for f in self.fragmentos]
        self.vetorizador = TfidfVectorizer(stop_words=None, lowercase=True)
        self.matriz_tfidf = self.vetorizador.fit_transform(corpus)
        print(f"[RAG] Base indexada com sucesso: {len(self.fragmentos)} blocos de texto carregados.")

    def buscar(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if self.matriz_tfidf is None or self.vetorizador is None:
            return []

        query_vec = self.vetorizador.transform([query])
        similaridades = cosine_similarity(query_vec, self.matriz_tfidf).flatten()

        indices_ordenados = np.argsort(similaridades)[::-1]
        resultados = []

        for idx in indices_ordenados[:top_k]:
            score = float(similaridades[idx])
            # Limiar mínimo de similaridade para considerar relevante
            if score >= 0.08:
                resultados.append({
                    "texto": self.fragmentos[idx]["texto"],
                    "fonte": self.fragmentos[idx]["fonte"],
                    "score": round(score, 4)
                })

        return resultados


# ==============================================================================
# BLOCO 4: FERRAMENTAS EXTERNAS & SÍNTESE COM GEMINI
# ==============================================================================
def consultar_pubmed_api(termo: str, max_resultados: int = 2) -> List[Dict[str, str]]:
    """Consulta a API do NCBI/PubMed via HTTP e devolve artigos relevantes."""
    url_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    termo_codificado = urllib.parse.quote(termo)
    url_esearch = f"{url_base}esearch.fcgi?db=pubmed&term={termo_codificado}&retmode=json&retmax={max_resultados}"

    try:
        req = urllib.request.Request(url_esearch, headers={"User-Agent": "FastAPI-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
            id_list = dados.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        id_str = ",".join(id_list)
        url_esummary = f"{url_base}esummary.fcgi?db=pubmed&id={id_str}&retmode=json"
        req_resumo = urllib.request.Request(url_esummary, headers={"User-Agent": "FastAPI-Agent/1.0"})

        with urllib.request.urlopen(req_resumo, timeout=5) as resposta_resumo:
            dados_resumo = json.loads(resposta_resumo.read().decode("utf-8"))
            resultados = []
            for pmid in id_list:
                item = dados_resumo.get("result", {}).get(pmid, {})
                titulo = item.get("title", "Título indisponível")
                resultados.append({
                    "fonte": f"PubMed PMID: {pmid}",
                    "titulo": titulo
                })
            return resultados
    except Exception as e:
        print(f"[Erro PubMed API]: {e}")
        return []


def gerar_resposta_sintese(pergunta: str, contexto: str, origem: str) -> str:
    """Gera síntese técnica estruturada utilizando o modelo gemini-3.8-flash."""
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
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048
            }
        )
        resposta = modelo.generate_content(
            prompt_completo,
            request_options={"timeout": 25}
        )
        return resposta.text.strip()
    except Exception as e:
        print(f"[Erro Gemini API]: {e}")
        return f"Falha na comunicação com a API do Gemini. Detalhes: {str(e)}"
        

def agente_decisor(pergunta: str, rag: MotorRAG) -> Dict[str, Any]:
    """Orquestrador do Agente: Decisão entre busca interna, externa e síntese."""
    trechos_locais = rag.buscar(pergunta, top_k=2)

    # Decisão 1: Se encontrar correspondência nos PDFs locais
    if trechos_locais:
        fontes = [t["fonte"] for t in trechos_locais]
        maior_score = trechos_locais[0]["score"]
        contexto_combinado = "\n\n---\n\n".join(
            [f"[Fonte: {t['fonte']}]: {t['texto']}" for t in trechos_locais]
        )

        resposta_final = gerar_resposta_sintese(
            pergunta=pergunta,
            contexto=contexto_combinado,
            origem="Documentos Técnicos Locais (Teses e Artigos)"
        )

        return {
            "origem": "RAG Local (Teses & Artigos)",
            "fontes": list(set(fontes)),
            "score": maior_score,
            "resposta": resposta_final
        }

    # Decisão 2: Acionamento da ferramenta externa (PubMed)
    artigos_externos = consultar_pubmed_api(pergunta, max_resultados=2)
    if artigos_externos:
        fontes = [a["fonte"] for a in artigos_externos]
        contexto_externo = "\n".join([f"- {a['titulo']} ({a['fonte']})" for a in artigos_externos])

        resposta_final = gerar_resposta_sintese(
            pergunta=pergunta,
            contexto=contexto_externo,
            origem="API Externa do PubMed / NCBI"
        )

        return {
            "origem": "API Externa (PubMed / NCBI)",
            "fontes": fontes,
            "score": 0.50,
            "resposta": resposta_final
        }

    # Decisão 3: Sem evidências
    return {
        "origem": "Sem Evidências",
        "fontes": [],
        "score": 0.0,
        "resposta": (
            "Não foram localizadas informações suficientes sobre essa consulta nos PDFs locais "
            "nem na busca automatizada do PubMed. Tente reformular a pergunta ou detalhar os termos."
        )
    }


# ==============================================================================
# BLOCO 5: INICIALIZAÇÃO DO MICROSSERVIÇO FASTAPI
# ==============================================================================
PASTA_DOCUMENTOS = Path(__file__).resolve().parent / "documentos"

app = FastAPI(
    title="Agente RAG de Biologia Molecular",
    version="1.0.0",
    description="Microsserviço de consulta técnica para vetores sintéticos e engenharia de Komagataella phaffii."
)

rag_service = MotorRAG(PASTA_DOCUMENTOS)


@app.get("/")
def home():
    return {
        "status": "online",
        "docs": "Acesse /docs para consultar a documentação interativa da API"
    }


@app.post("/perguntar", response_model=RespostaConsulta)
def perguntar_ao_agente(requisicao: RequisicaoConsulta):
    resultado = agente_decisor(requisicao.pergunta, rag_service)
    return RespostaConsulta(
        usuario_id=requisicao.usuario_id,
        pergunta=requisicao.pergunta,
        origem_dados=resultado["origem"],
        fontes_consultadas=resultado["fontes"],
        resposta=resultado["resposta"],
        score_confianca=resultado["score"]
    )