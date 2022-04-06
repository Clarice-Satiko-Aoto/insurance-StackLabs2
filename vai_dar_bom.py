# -*- coding: utf-8 -*-
"""
@author: @Clarice_Aoto
"""

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import PIL
from PIL import Image



st.set_page_config(
    page_title="Koalas",
    page_icon="🐨",
    layout="centered",
    initial_sidebar_state='auto',
    menu_items=None)


paginas = ['Sistema', 'Insurance Prediction', "Roadmap",'Equipe', 'Agradecimentos']

###### SIDE BAR ######
col1, col2, col3 = st.sidebar.columns([1, 3, 1])
with col2:
    image1 = Image.open('./imagens/Koalas_B2.png')
    st.image(image1, width=120)

    pagina = st.sidebar.radio("Navegação", paginas)



###### Sistema ######
if pagina== "Sistema":
    st.subheader('Sistema - Análise de clientes')

    uploaded_file = st.file_uploader("faça o upload do arquivo:")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        print(df) # checar a saída no terminal
        # predição dos novos dados
        loaded_model = pickle.load(open('model_lr04.06.1.pkl', 'rb'))
        #print(loaded_model)
        y_pred = loaded_model.predict(df)
        resultado = collections.Counter(y_pred)
        count_resultado = len(y_pred)
        st.write("Total de clientes analisados: ", count_resultado)

        interessados = (y_pred == 1).sum()
        desinteressados = (y_pred == 0).sum()
        percentual_pos = (count_resultado/interessados).round(2)
        percentual_neg = (count_resultado/desinteressados).round(2)

        st.write("Total de interessados: ", interessados, '(', percentual_pos,'%)')
        st.write("Total de desinteressados: ", desinteressados, '(', percentual_neg, '%)')
      
        vintage = df['Vintage'].mean().round(2)
        st.write('Média de período em dias que os clientes estão/foram credenciados: ' ,vintage)
        premio = df['Annual_Premium'].mean().round(2)
        st.write('Média do prêmio anual dos clientes: ', premio)

        idade_media = pd.DataFrame(df['Age'].value_counts())
        st.bar_chart(idade_media)
        # with column_sis2:
        # st.write('A empresa possui ', count_resultado, 'clientes na carteira, \
        #     e irá tomar as priemeiras ações de marketing com ', interessados,'interessados.') 
            

        insured = pd.DataFrame(df['Previously_Insured'].value_counts())
        st.bar_chart(insured)

        vehicle= pd.DataFrame(df['Vehicle_Damage'].value_counts())
        st.bar_chart(vehicle)        
        



###### Insurance Prediction ######
if pagina== "Insurance Prediction":
    st.subheader('Insurance Prediction Web App')

    # loading the saved model
    loaded_model = pickle.load(open('model_lr04.06.1.pkl', 'rb'))

    column_1, column_2 = st.columns(2)
    with column_1:
        Age = st.number_input('Age', min_value=18, max_value=85)
        Vehicle_Damage = st.number_input('Vehicle_Damage (0: NO, 1:YES)', min_value=0, max_value=1)
        Vintage = st.number_input('Vintage (days)',min_value=0)
    with column_2:
        Previously_Insured = st.number_input('Previously_Insured (0: NO, 1:YES)',min_value=0, max_value=1)
        Annual_Premium = st.number_input('Annual_Premium (USD)')


    diagnosis = [Age, Previously_Insured, Vehicle_Damage, Annual_Premium, Vintage]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(diagnosis)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # st.success(diagnosis)

    if st.button('Result'):
        prediction = loaded_model.predict(input_data_reshaped)

        if (prediction[0] == 0):
            st.write('Cliente não tem interesse')
        else:
            st.write('Cliente interessado!')
            st.balloons()
    


    
    
###### Jornada do produto ######
if pagina== "Roadmap":
    st.header('**Roadmap do projeto**')

    st.subheader("**Engenharia de dados**")
    dados = Image.open("./imagens/Eng_Dados.png")
    st.image(dados, width=700)

    st.subheader('**Análise exploratória**')
    st.markdown('Abaixo alguns gráficos da fase de análise exploratória:')
    df = pd.read_csv('./new_start/train.csv')

    idade_carro = pd.DataFrame(df['Vehicle_Age'].value_counts())
    st.bar_chart(idade_carro)


    sinistro = pd.DataFrame(df['Previously_Insured'].value_counts())
    st.bar_chart(sinistro)

    idade = pd.DataFrame(df['Age'].value_counts())
    st.line_chart(idade)

    canal = pd.DataFrame(df['Policy_Sales_Channel'].value_counts())
    st.line_chart(canal)







    st.write('*Principais insights*')
    st.markdown('* Perfil dos clientes interessados – Maioria homens, média de 40 anos, aproximadamente 6 meses na base, não tiveram seguro anteriormente mas já tiveram algum tipo de sinistro, idade do veículo entre 1 e 2 anos.')
    st.markdown('* Porcentagem de interessados total – 12,26%')
    st.markdown('* Distribuição da idade dos consumidores – Aceitação em média 40 anos e rejeição em média 35 anos')
    st.markdown('* Relação entre prêmio e idade do cliente – não há correlação')
    st.markdown('* Diferença entre gêneros no interesse veicular – Homens possuem maior interesse')
    st.markdown('* Quantidade de clientes que já tiveram seguro veicular – dos entrevistados 174.628')
    st.markdown('* Idade do veículo interfere no interesse do seguro – possuem maior interesse os donos de veículos entre 1 e 2 anos.')
    st.markdown('* Cliente com carros sinistrados tem maior propensão em adquirir seguro veicular – não, clientes com carros sinistrados na média não tem interesse em adquirir novo seguro.')
    st.markdown('* Tempo na base de seguro saúde influencia no interesse do seguro veicular – não se nota diferença entre interessados e não interessados')
    
    
    st.subheader('**Modelo preditivo**')
    code0= """# Analisando a importância de cada feature
#instanciando
RF_Class = RandomForestClassifier(random_state=123)
RF_Class.fit(X,y)


#buscando as melhores features
feature_imp = pd.Series(RF_Class.feature_importances_,index=Xu.columns).sort_values(ascending=False)

# Selecionando as features de maior importância.
features_selected = []
for feature,importance in feature_imp.iteritems():
    if importance > 0.03:
        print(f'{feature}: {round(importance * 100)}%')
        features_selected.append(feature) """
    
    code1 =""" #Instanciando o modelo
lr = LogisticRegression(random_state=123)
#treinando o modelo 
lr.fit(X_treino_u, y_treino_u)
#predizendo os valores
y_pred_lr_u = lr.predict(X_teste_u)
#imprindo Relatorio de classificação
print(classification_report(y_teste_u, y_pred_lr_u))
#Matriz de confusão
print(pd.crosstab(y_teste_u, y_pred_lr_u, rownames=['Real'],
    colnames=['Predito'], margins=True))"""

    st.markdown('Para a produção do modelo preditivo, analisamos as principais features do dataset de treino\
        e para o modelo final do nosso MVP, utilzamos as colunas Annual_premiun, Vintage, Vehicle_Damage, \
            Previously_insured e Age para demonstrar aos *stakeholders* a eficácia do nosso modelo preditivo.')
    st.code(code0, language='python')
    st.markdown('Saida:')
    feature = Image.open("./imagens/feat_import.png")
    st.image(feature, width = 150)
    st.markdown('Foi utilizada o *Logistic Regression* que determina a probabilidade de um evento acontecer.\
        Ele mostra a relação entre os recursos e, em seguida, calcula a probabilidade de um determinado resultado ')
    st.markdown('Seguem o código utilizado e a saída:')
    st.code(code1, language='python')
    resultado = Image.open("./imagens/saida_modelo.png")
    st.image(resultado, width=300)

    
    st.subheader('**Conclusão**')
    st.markdown('texto')


###### Equipe ######
if pagina== "Equipe":
    st.subheader('Squad Koalas')
# membro1
    col1,col2,col3 = st.columns([1,3,2])
    col1,col2,col3 = st.columns([1,3,2])
    col1,col2,col3 = st.columns([1,3,2])
    with col1:
            image3 = Image.open("./imagens/Clarice.png")
            st.image(image3, width=100)
            col2.markdown('**Clarice Satiko Aoto**')
            col2.write("Back-end | UI Designer Jr.")
            col2.write("[Linkedin](https://www.linkedin.com/in/claricesatikoaoto-bi-python-ux/)")

    
    # membro2
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            image3 = Image.open("./imagens/isis.png")
            st.image(image3, width=100)
            col2.markdown('**Isis Karina de Souza**')
            col2.write("Data Engineer Jr.")
            col2.write("[Linkedin](https://www.linkedin.com/in/isiskarina/)")
    
   # membro3
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            image3 = Image.open("./imagens/sarah.png")
            st.image(image3, width=100)
            col2.markdown('**Sarah David Müzel**')
            col2.write("Data Analystic | Data Engineer")
            col2.write("[Linkedin](https://www.linkedin.com/in/sarah-david-m%C3%BCzel-05525356/)")    
    
    
    # membro4
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    col1,col2,col3 = st.columns([1,2,3])
    with col1:
            image3 = Image.open("./imagens/vivian.png")
            st.image(image3, width=100)
            col2.markdown('**Vivian Andrade**')
            col2.write("Data Scientist Pleno")
            col2.write("[Linkedin](https://www.linkedin.com/in/vivian-cientista-de-dados/)")


##### Agradecimentos #####
if pagina== "Agradecimentos":
    st.subheader('Agradecimentos')

    st.markdown('Como diz um velho ditado : **“Quem tem padrinho não morre pagão”**.')
    st.markdown('Nunca se fez tão real e atual essa parábola, pois com a Koalas Squad não poderia ser diferente.')

    st.markdown('Sem a rede de apoio que tivemos neste projeto, nada seria possível sem esses padrinhos!.')

    st.markdown('Agradecemos:')

    st.markdown(' * [**Eduardo Moraes**](https://www.linkedin.com/in/eduardo-moraes-ds/) pela amizade, generosidade, atenção e carinho. Qando o nosso amigo Google não nos ajudava, estava lá o Eduardo para nos salvar. Obrigada, Edu!')
    st.markdown(' * [**Alberlando Herculano (Albert)**](https://www.linkedin.com/in/alberlandoherculano/) por nos ensinar o que é a vida real. Agora sabemos o que é  a  expectativa e a realidade. Obrigada!')
    st.markdown('* [**Matth**](https://www.linkedin.com/in/matheus-marques-/) que chegou de mansinho, tirando suas dúvidas e foi conquistando nosso coração com suas palavras de incentivo, com sua amizade e disponibilidade para emprestar o ombro nos momentos de pânico. Obrigada, Matth!')
    st.markdown('* [**Enzo Niro**](https://www.linkedin.com/in/enzo-niro-59a11537/) que lá no comecinho nos ajudou na arquitetura e engenharia da AWS! E quando ele não sabia responder, se disponibiliou em perguntar ao chefe dele. Obrigada, Enzo e chefe do Enzo!')
    st.markdown('* [** Gabriel Souza**](https://www.linkedin.com/in/gabriel-sousa/), ah, Gabis... você é incrível! Obrigada!')
    st.markdown('* Aos veteranos da koalas Squad. Mesmo não estando juntos, temos a certeza da torcida de vocês! Obrigada, [**Peterson**](https://www.linkedin.com/in/peterson-rosa-silva/), [**Marcão**](https://www.linkedin.com/in/mcosta7/) e [**Octávio**](https://www.linkedin.com/in/octavio-oliveira-56974178/)!')
    st.markdown('* [**Luiz Carlos de Jesus Júnior**](https://www.linkedin.com/in/lcjr86/) por ser nosso amigo e guia, que nos manteve centradas em nossas tarefas. Obrigada!')
    st.markdown('* [**Clarice Satiko Aoto**](https://www.linkedin.com/in/claricesatikoaoto-bi-python-ux/), sim ela é membro da squad, mas de comum acordo não poderíamos deixar de fazer um agradecimento especial a essa pessoa linda por dentro e por fora. Mesmo no sacrifício ela esteve presente. Até no silêncio ela nos ajudava, pois nos colocava a pensar : "Como Clarice faria?" Amiga de todas nós, obrigada!')
    st.markdown('* Aos nossos mestres [**Felipe**](https://www.linkedin.com/in/felipesf/) e [**Rodrigo**](https://www.linkedin.com/in/rodrigo-santana-ferreira-0ab041128/) pelos ensinamentos,  parceria, amizade e paciência, afinal, ter uma squad tão barulhenta não é para qualquer um não. Obrigada!')
    st.markdown('A todos os amigos e familiares que torcem pelo nosso sucesso! Agradecemos também a todas as outras squads que enfrentaram dificuldades e estão na mesma jornada! Todos vocês ativam a nossa sede por mais conhecimento! Estamos na torcida por vocês!')

    st.markdown('Um forte abraço,')

    st.markdown('**Koalas Squad**')