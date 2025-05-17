%pip -q install google-genai google-adk

import os
from google.colab import userdata
from datetime import date
import textwrap
import warnings
from IPython.display import display, Markdown, HTML

from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

warnings.filterwarnings("ignore")

# Configuração da API Key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
client = genai.Client()
MODEL_ID = "gemini-2.0-flash" # Podemos ajustar o modelo conforme a necessidade

def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text
                    final_response += "\n"
    return final_response

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def agente_identificador_necessidades(problema_comunidade, area_comunidade):
    identificador = Agent(
        name="agente_identificador_necessidades",
        model=MODEL_ID,
        instruction=f"""
        Você é um especialista em analisar o seguinte problema relatado na comunidade de {area_comunidade}: "{problema_comunidade}".
        Sua tarefa é usar a ferramenta de busca do Google (google_search) para entender melhor esse problema específico no contexto de {area_comunidade}.
        Busque por informações sobre:
        - Causas comuns desse tipo de problema.
        - Impactos na comunidade.
        - Iniciativas semelhantes que foram implementadas com sucesso em outras localidades.
        Identifique até 3 aspectos chave do problema que precisam ser abordados para encontrar soluções eficazes na comunidade de {area_comunidade}.
        """,
        description="Agente que aprofunda a compreensão do problema da comunidade.",
        tools=[google_search]
    )
    entrada = f"Problema: {problema_comunidade} na comunidade de {area_comunidade}. Causas, impactos, soluções em outras comunidades."
    analise = call_agent(identificador, entrada)
    return analise

def agente_mapeador_recursos(area_comunidade, analise_problema):
    mapeador = Agent(
        name="agente_mapeador_recursos",
        model=MODEL_ID,
        instruction=f"""
        Você é um especialista em mapear os recursos disponíveis na comunidade de {area_comunidade} que podem ajudar a solucionar o problema,
        cujos aspectos chave foram identificados como: {analise_problema}.
        Sua tarefa é usar a ferramenta de busca do Google (google_search) para encontrar informações sobre:
        - Voluntários com habilidades relevantes para o problema.
        - Serviços locais que podem oferecer suporte ou soluções.
        - Organizações não governamentais e iniciativas comunitárias focadas em problemas semelhantes ou que atuam em {area_comunidade}.
        - Espaços ou equipamentos que poderiam ser utilizados para ações de solução.
        Liste os tipos de recursos relevantes e, se possível, exemplos específicos encontrados na busca para a comunidade de {area_comunidade}.
        """,
        description="Agente que mapeia recursos comunitários relevantes para o problema.",
        tools=[google_search]
    )
    entrada = f"Recursos disponíveis em {area_comunidade} para solucionar o problema (aspectos chave: {analise_problema}): voluntários, serviços locais, ONGs, espaços."
    recursos_mapeados = call_agent(mapeador, entrada)
    return recursos_mapeados

def agente_gerador_solucoes(area_comunidade, problema, analise_problema, recursos_mapeados):
    gerador = Agent(
        name="agente_gerador_solucoes",
        model=MODEL_ID,
        instruction=f"""
        Você é um especialista em gerar soluções criativas e práticas para o problema "{problema}" na comunidade de {area_comunidade},
        considerando a análise do problema: {analise_problema} e os recursos mapeados: {recursos_mapeados}.
        Sua tarefa é propor pelo menos 3 soluções distintas e viáveis que a própria comunidade poderia implementar, utilizando os recursos disponíveis.
        Cada solução deve incluir:
        - Uma breve descrição da ação.
        - Os principais recursos necessários (identificados pelo agente mapeador ou outros que você considere relevantes).
        - Os potenciais benefícios para a comunidade.
        Seja criativo e pense em soluções que promovam a colaboração e o engajamento dos moradores.
        """,
        description="Agente que gera soluções para o problema da comunidade.",
        tools=[] # Não necessariamente precisa de busca, mas pode ser útil para inspiração
    )
    entrada = f"Gerar 3 soluções para o problema '{problema}' em {area_comunidade}, considerando a análise: {analise_problema} e os recursos: {recursos_mapeados}."
    solucoes_propostas = call_agent(gerador, entrada)
    return solucoes_propostas

def agente_avaliador_solucoes(problema, area_comunidade, solucoes_propostas):
    avaliador = Agent(
        name="agente_avaliador_solucoes",
        model=MODEL_ID,
        instruction=f"""
        Você é um especialista em avaliar a viabilidade e o potencial impacto das seguintes soluções propostas para o problema "{problema}" na comunidade de {area_comunidade}:
        {solucoes_propostas}.
        Para cada solução, sua tarefa é analisar:
        - A probabilidade de sucesso com os recursos disponíveis.
        - Os potenciais desafios ou obstáculos para a implementação.
        - O impacto esperado na resolução do problema e no bem-estar da comunidade.
        Apresente uma breve avaliação para cada solução, destacando seus pontos fortes e fracos.
        """,
        description="Agente que avalia a viabilidade e o impacto das soluções propostas.",
        tools=[] # Não precisa de busca para avaliar as soluções apresentadas
    )
    entrada = f"Avaliar as soluções para o problema '{problema}' em {area_comunidade}: {solucoes_propostas} (viabilidade, desafios, impacto)."
    avaliacao_solucoes = call_agent(avaliador, entrada)
    return avaliacao_solucoes

print("🤝 Bem-vindo ao Sistema 'Elo Comunitário' 🤝")

problema = input("❓ Qual é o principal problema que você gostaria de resolver em sua comunidade? ")
area = input("📍 Em qual localidade/bairro específico esse problema ocorre? ")

if not problema or not area:
    print("Por favor, informe o problema e a localização.")
else:
    print(f"\n🤔 Analisando o problema '{problema}' em {area}...")
    analise = agente_identificador_necessidades(problema, area)
    print("\n--- 🔍 Análise do Problema ---\n")
    display(to_markdown(analise))
    print("--------------------------------------------------------------")

    print(f"\n🗺️ Mapeando os recursos disponíveis em {area} para o problema...")
    recursos = agente_mapeador_recursos(area, analise)
    print("\n--- 💡 Recursos Mapeados ---\n")
    display(to_markdown(recursos))
    print("--------------------------------------------------------------")

    print("\n💡 Gerando soluções para o problema...")
    solucoes = agente_gerador_solucoes(area, problema, analise, recursos) # Correção aqui: passando 'analise'
    print("\n--- ✨ Soluções Propostas ---\n")
    display(to_markdown(solucoes))
    print("--------------------------------------------------------------")

    print("\n🧐 Avaliando a viabilidade das soluções...")
    avaliacao = agente_avaliador_solucoes(problema, area, solucoes)
    print("\n--- 📊 Avaliação das Soluções ---\n")
    display(to_markdown(avaliacao))
    print("--------------------------------------------------------------")

    print("\n🎉 O sistema 'Elo Comunitário' gerou algumas ideias e avaliações para ajudar a resolver o problema na sua comunidade!")