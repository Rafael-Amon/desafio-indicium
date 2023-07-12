import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

carros = pd.read_parquet('./data/carros.parquet')

# Mapear Colunas com respostas únicas para 1
carros['dono_aceita_troca'] = carros['dono_aceita_troca'].map({'Aceita troca': 1})
carros['veiculo_único_dono'] = carros['veiculo_único_dono'].map({'Único dono': 1})
carros['revisoes_concessionaria'] = carros['revisoes_concessionaria'].map({'Todas as revisões feitas pela concessionária': 1})
carros['ipva_pago'] = carros['ipva_pago'].map({'IPVA pago': 1})
carros['veiculo_licenciado'] = carros['veiculo_licenciado'].map({'Licenciado': 1})
carros['garantia_de_fábrica'] = carros['garantia_de_fábrica'].map({'Garantia de fábrica': 1})
carros['revisoes_dentro_agenda'] = carros['revisoes_dentro_agenda'].map({'Todas as revisões feitas pela agenda do carro': 1})

# Preencher valores NaN com 0
carros['dono_aceita_troca'].fillna(0, inplace=True)
carros['veiculo_único_dono'].fillna(0, inplace=True)
carros['revisoes_concessionaria'].fillna(0, inplace=True)
carros['ipva_pago'].fillna(0, inplace=True)
carros['veiculo_licenciado'].fillna(0, inplace=True)
carros['garantia_de_fábrica'].fillna(0, inplace=True)
carros['revisoes_dentro_agenda'].fillna(0, inplace=True)

def buscar_categorias():
    
    marcas = list(carros['marca'].unique())
    cambios = list(carros['cambio'].unique())
    tipos_carro = list(carros['tipo'].unique())
    blindados = list(carros['blindado'].unique())
    cores = list(carros['cor'].unique())
    estados_vendedor = list(carros['estado_vendedor'].unique())
    anunciantes = list(carros['anunciante'].unique())
    veiculos_único_dono = list(carros['veiculo_único_dono'].unique())
    garantias_de_fábrica = list(carros['garantia_de_fábrica'].unique())

    return marcas, cambios, tipos_carro, blindados, cores, estados_vendedor, anunciantes, veiculos_único_dono, garantias_de_fábrica


def predicao_carros(caracteristicas):
    '''
    Função para realizar a predição de carros
    '''
    
    ordem_colunas = ['marca', 'ano_de_fabricacao', 'ano_modelo', 'hodometro', 'cambio',
                     'num_portas', 'tipo', 'blindado', 'cor', 'tipo_vendedor',
                     'estado_vendedor', 'anunciante', 'dono_aceita_troca',
                     'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago',
                     'veiculo_licenciado', 'garantia_de_fábrica', 'revisoes_dentro_agenda']

    df_caracteristicas = pd.DataFrame(columns=ordem_colunas)
    df_caracteristicas.loc[0] = caracteristicas

    colunas_categoricas = [coluna for coluna in colunas_categoricas if coluna in df_caracteristicas.columns]

    ohe = OneHotEncoder(sparse=False, drop='first')

    for coluna_categorica in colunas_categoricas:
        ohe.fit(carros[[coluna_categorica]])
        colunas_ohe_teste = ohe.transform(df_caracteristicas[[coluna_categorica]])

        categorias_ohe = ohe.categories_[0][1:]

        for indice, nome_categoria in enumerate(categorias_ohe):
            df_caracteristicas[nome_categoria] = colunas_ohe_teste[:, indice]

        df_caracteristicas = df_caracteristicas.drop(coluna_categorica, axis=1)

    colunas_numericas = [coluna for coluna in ordem_colunas if coluna not in colunas_categoricas]

    scaler = StandardScaler()

    for coluna_numerica in colunas_numericas:
        scaler.fit(carros[[coluna_numerica]])
        df_caracteristicas[coluna_numerica] = scaler.transform(df_caracteristicas[[coluna_numerica]])

    modelo = joblib.load('./models/xgb_model.pkl')
    preco = modelo.predict(df_caracteristicas)

    return preco


# Criando a interface

cabecalho = st.container()
features = st.container()
resultado = st.container()

with cabecalho:
    st.image('./img/header.png', width=600)
    st.write('\n')
    st.title("Previsão de Preços de Carros")

with features:
    marcas, cambios, tipos_carro, blindados, cores, estados_vendedor, anunciantes, veiculos_único_dono, garantias_de_fábrica = buscar_categorias()

    st.sidebar.title('Informe as características do carro')

    ano_de_fabricacao = st.sidebar.slider("Ano de Fabricação", min_value=2003, max_value=2021)
    ano_modelo = st.sidebar.slider("Ano do Modelo", min_value=2003, max_value=2021)
    hodometro = st.sidebar.slider("KMs rodados (em milhares)", min_value=0., max_value=200.) * 1000
    num_portas = st.sidebar.slider("Portas", min_value=2., max_value=6.)

    marca = st.sidebar.selectbox("Qual a marca do carro?", options=marcas)
    tipo = st.sidebar.selectbox("Qual o tipo do carro?", options=tipos_carro)
    cambio = st.sidebar.selectbox("Qual o câmbio?", options=cambios)
    garantia_de_fábrica = st.sidebar.selectbox("Carro está na garantia? (0-Não 1-Sim)", options=garantias_de_fábrica)
    #blindado = st.sidebar.selectbox("Carro é blindado? (0-Não 1-Sim)", options=blindados)
    cor = st.sidebar.selectbox("Qual a cor do carro?", options=cores)
    estado_vendedor = st.sidebar.selectbox("Qual o estado (UF) do carro?", options=estados_vendedor)
    anunciante = st.sidebar.selectbox("Quuem é o anunciante?", options=anunciantes)
    tipo_vendedor = 'PF'
    dono_aceita_troca = '0'
    veiculo_único_dono = '0'
    revisoes_concessionaria = '0'
    ipva_pago = '1'
    veiculo_licenciado = '1'
    revisoes_dentro_agenda = '0'
    blindado = 'N'



    if st.sidebar.button("Calcular preço"):
        caracteristicas = [marca,
                           ano_de_fabricacao,
                           ano_modelo,
                           hodometro,
                           cambio,
                           num_portas,
                           tipo,
                           blindado,
                           cor,
                           tipo_vendedor,
                           estado_vendedor,
                           anunciante,
                           dono_aceita_troca,
                           veiculo_único_dono,
                           revisoes_concessionaria,
                           ipva_pago,
                           veiculo_licenciado,
                           garantia_de_fábrica,
                           revisoes_dentro_agenda]

        print(caracteristicas)
        preco_predito = predicao_carros(caracteristicas)

        print(f"*** PRECO: {preco_predito}")

        with cabecalho:
            st.text(f"PRECO DO CARRO: {preco_predito[0]:.2f} reais")