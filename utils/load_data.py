import pandas as pd


# def get_df_skins(file_path, year):
#     """
#     Coleta o dataframe e retorna somente as observacoes com resultado com as doenças que serao estudadas
#     """
#     df_skin = pd.DataFrame()
#     nomes_registros = ["São Gabriel da Palha", "Itarana", "Vila Pavão", "Paraju", "Santa Maria do Jetiba"]
#     if year == 2021:
#         df = pd.read_excel(file_path, sheet_name=nomes_registros)
#         for nome_registro in nomes_registros:
#             df_skin = pd.concat([df_skin, pd.DataFrame(df[nome_registro])])
#     else:
#         df_skin = pd.read_excel(file_path)

#     df_skin = df_skin \
#                 .loc[pd.notnull(df_skin["resultado"])] \
#                 .rename(columns={"Unnamed: 0": "index_sus"}) \
#                 .drop(['Unnamed: 0.1', 'NÃºmero de sÃ©rie do instrumento', 
#                 'Temperatura', 'Notas', 'Carimpo de Tempo'], axis=1)

#     # df_skin = df_skin.loc[~df_skin['index_sus'].str.contains('-1.sam')]
#     return df_skin


def get_df_skins(file_path):
    """
    Coleta o dataframe e retorna somente as observacoes com resultado com as doenças que serao estudadas
    """
    df_skin = pd.read_excel(file_path)
    df_skin = df_skin.loc[pd.notnull(df_skin["resultado"])]
    df_skin = df_skin.loc[~df_skin['index_sus'].str.contains('-0-')].drop('index', axis=1)
    return df_skin.reset_index(drop=True)

def get_df_skins_vv(file_path):
    """
    Coleta o dataframe e retorna somente as observacoes com resultado com as doenças que serao estudadas
    """
    df_skin = pd.read_excel(file_path)
    df_skin = df_skin.loc[pd.notnull(df_skin["ysample"])]
    df_skin = df_skin.loc[~df_skin['COD'].str.contains('-00-')]
    return df_skin.reset_index(drop=True)


# melanoma, carcinoma basocelular, carcinoma espinocelular(carcinoma de celulas escamosas) -> malignos
# nevo, ceratose actinica, ceratose seborreica -> benignos
def get_labels(df_skin):
    """
    Coleta os labels que serao utilizados da variavel resultado
    """
    result_mal = ["melanoma", "carcinoma basocelular", "carcinoma de celulas escamosas"]
    result_ben = ["nevo", "ceratose actinica", "ceratose seborreica"]
    for res in result_ben:
        df_skin.loc[df_skin["resultado"].str.contains(res), "resultado"] = res
    df_ben = df_skin.loc[df_skin["resultado"].isin(result_ben)]
    df_ben.loc[:, "label"] = "benigno"

    for res in result_mal:
        df_skin.loc[df_skin["resultado"].str.contains(res), "resultado"] = res
    df_mal = df_skin.loc[df_skin["resultado"].isin(result_mal)]
    df_mal.loc[:, "label"] = "maligno"

    df = pd.concat([df_ben, df_mal])
    print(f"Dimensao do dataframe final: {df.shape}")
    return df.reset_index(drop=True)


def get_df_skins_vv(file_path):
    """
    Coleta o dataframe e retorna somente as observacoes com resultado com as doenças que serao estudadas
    """
    df_skin = pd.read_excel(file_path)
    df_skin = df_skin.loc[pd.notnull(df_skin["ysample"])]
    df_skin = df_skin.loc[~df_skin['COD'].str.contains('-00-')]
    df_skin = df_skin.rename(columns={"COD": "index_sus"})
    # df = get_df_skins_vv("../../data/pad_Vvalerio2206.xlsx")
    df = df_skin
    resultado = df.ysample
    df = df.drop('ysample', axis=1)
    df['resultado'] = resultado
    df['resultado'] = df.resultado.map({'ACK': "ceratose actinica", 'SEK': "ceratose seborreica", 
                                        'NEV': "nevo", 'MEL': "melanoma"})
    # df = df.rename(columns={'COD': 'index_sus'}).dropna()
    return df.reset_index(drop=True)

def get_labels(df_skin):
    """
    Coleta os labels que serao utilizados da variavel resultado
    """
    df_skin = df_skin.dropna(subset=["resultado"])
    result_mal = ["melanoma", "carcinoma basocelular", "carcinoma espinocelular"]
    result_ben = ["nevo", "ceratose actinica", "ceratose seborreica"]
    for res in result_ben:
        df_skin.loc[df_skin["resultado"].str.contains(res), "resultado"] = res
    df_ben = df_skin.loc[df_skin["resultado"].isin(result_ben)]
    df_ben.loc[:, "label"] = "benigno"

    for res in result_mal:
        df_skin.loc[df_skin["resultado"].str.contains(res), "resultado"] = res
    df_mal = df_skin.loc[df_skin["resultado"].isin(result_mal)]
    df_mal.loc[:, "label"] = "maligno"

    df = pd.concat([df_ben, df_mal])
    print(f"Dimensao do dataframe final: {df.shape}")
    return df.reset_index(drop=True)

def get_labels_complete(df_skin):
    """
    Coleta os labels que serao utilizados da variavel resultado
    """
    df_skin = df_skin.dropna(subset=["resultado"])

    result_ben = ["nevo", "ceratose actinica", "ceratose seborreica"]
    result_mal = ["melanoma", "carcinoma basocelular", "carcinoma espinocelular"]
    df_skin.loc[df_skin["resultado"].isin(result_ben), "label"] = "benigno"
    df_skin.loc[df_skin["resultado"].isin(result_mal), "label"] = "maligno"


    # df_ben = df_skin.loc[df_skin["resultado"].isin(result_ben), "label"]
    # df_ben.loc[:, "label"] = "benigno"

    # df_mal = df_skin.loc[df_skin["resultado"].isin(result_mal), "label"]
    # df_mal.loc[:, "label"] = "maligno"

    # df = pd.concat([df_ben, df_mal])
    print(f"Dimensao do dataframe final: {df_skin.shape}")
    return df_skin


def get_means_spectres():
    df = pd.read_csv("../../data/data_complete.csv", index_col=0).reset_index(drop=True)
    df_split = df['index_sus'].str.split('-', expand=True)
    df['index_sus'] = df_split[0] + '-' + df_split[1] + '-'
    df_mean = df.groupby('index_sus', as_index=False).mean()
    df_last = df.groupby('index_sus', as_index=False).last()
    df_mean['resultado'] = df_last['resultado']
    df_mean['label'] = df_last['label']
    df_mean = df_mean.loc[~df_mean.index_sus.str.contains('-0-')].reset_index(drop=True)
    return df_mean
