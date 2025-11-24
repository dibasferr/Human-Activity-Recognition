#FICHEIRO UTILIZADO PARA GUARDAR IDEIAS ABSURDAS DE CODIGO 




#3.1
def split_data_intraSubj(subj):
    #Escolher o K
    #Fazer para varios K's e determinar o melhor K 
    
    for i in range(0,6): #6 splits por pessoa
        
        #NOTA: COM random_state FIXO A DISTRIBUIÇÃO É SEMPRE IGUAL
        # penultima coluna refere-se a atividade e a ultima é a pessoa em causa
        
        mask = embeddings[:,-1] == subj  # dados dessa pessoa
        X_subj = embeddings[mask]
        y_subj = embeddings[:,-2][mask]
        
        ############################################ SPLIT ##############################################
        # 60% treino, 40% resto
        X_train_s, X_temp, y_train_s, y_temp = train_test_split(
            X_subj, y_subj, test_size=0.4, random_state=42+i, shuffle=True)
        
        # 20/20 split do restante
        X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42+i, shuffle=True)
        
        ############################################ SPLIT FEITO ##############################################
        # X_train_s  X_val_s X_test_s 
        # y_train_s  y_val_s y_test_s
        ########################################## TREINO EMBEDDINGS SEM REDUCAO #################################
       
        #FAZER TREINO
        ########################################## TREINO EMBEDDINGS COM PCA #################################
        
        # índice da primeira vez que a variância acumulada >= 0.90 by default
        num_dim=0
        #alterado para a segunda meta
        pesos_PCA= aplicar_pca(X_train, 0.90)
        num_dim = np.argmax(pesos_PCA >= 0.90) + 1
        
        pca_reduced = PCA(n_components=num_dim)
        
        X_train_s = pca_reduced.fit_transform(X_train_s)
        X_val_s = pca_reduced.fit_transform(X_val_s)
        X_test_s = pca_reduced.fit_transform(X_test_s)
        
        #FAZER TREINO
        ########################################################################################################
        
        
        ####################################### EMBEDDING RELIEFF ######################################################
        pesos= reliefF(X_subj,5) #5 numero provisorio
        top15= quinze_melhores_features(pesos)
        X_reduzido = X_subj[:, top15]
        
        mask = X_reduzido[:,-1] == subj  # dados dessa pessoa
        
        X_subj = X_reduzido[mask]
        y_subj = X_reduzido[:,-2][mask]

        # 60% treino, 40% resto
        X_train_s, X_temp, y_train_s, y_temp = train_test_split(
            X_subj, y_subj, test_size=0.4, random_state=42+i, shuffle=True)
        
        # 20/20 split do restante
        X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42+i, shuffle=True)
        #FAZER TREINO
        
        ############################################ EBEDDING PCA #####################################################
        mask = embedding[:,-1] == subj  # dados dessa pessoa
    
        X_subj = embedding[mask]
        
        X_reduced= aplicar_pca(X_subj,0.90)
        
        mask = X_reduced[:,-1] == subj  # dados dessa pessoa
        
        X_subj = X_reduced[mask]
        y_subj = X_reduced[:,-2][mask]
        
        # 60% treino, 40% resto
        X_train_s, X_temp, y_train_s, y_temp = train_test_split(
            X_subj, y_subj, test_size=0.4, random_state=42+i, shuffle=True)
        
        # 20/20 split do restante
        X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42+i, shuffle=True)
        #FAZER TREINO
        
       
        ############################################ SPLIT FEATURES ###################################################
        mask = vetor_features[:,-1] == subj  # dados dessa pessoa
    
    
        X_subj = vetor_features[mask]
        y_subj = vetor_features[:,-2][mask]
        
        # 60% treino, 40% resto
        X_train_s, X_temp, y_train_s, y_temp = train_test_split(
            X_subj, y_subj, test_size=0.4, random_state=42+i, shuffle=True)
        
        # 20/20 split do restante
        X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42+i, shuffle=True)
        #FAZER TREINO
        
        
        ####################################### FEATURES RELIEFF ########################################################
        pesos= reliefF(X_subj,5) #5 numero provisorio
        top15= quinze_melhores_features(pesos)
        X_reduzido = X_subj[:, top15]
        
        mask = X_reduzido[:,-1] == subj  # dados dessa pessoa
        
        X_subj = X_reduzido[mask]
        y_subj = X_reduzido[:,-2][mask]

        # 60% treino, 40% resto
        X_train_s, X_temp, y_train_s, y_temp = train_test_split(
            X_subj, y_subj, test_size=0.4, random_state=42+i, shuffle=True)
        
        # 20/20 split do restante
        X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42+i, shuffle=True)
        #FAZER TREINO
        
        #################################################################################################################
    

        ######################################## SPLIT FEATURES PCA #####################################################
        mask = vetor_features[:,-1] == subj  # dados dessa pessoa
    
        X_subj = vetor_features[mask]
        
        X_reduced= aplicar_pca(X_subj,0.90)
        
        mask = X_reduced[:,-1] == subj  # dados dessa pessoa
        
        X_subj = X_reduced[mask]
        y_subj = X_reduced[:,-2][mask]

        # 60% treino, 40% resto
        X_train_s, X_temp, y_train_s, y_temp = train_test_split(
            X_subj, y_subj, test_size=0.4, random_state=42+i, shuffle=True)
        
        # 20/20 split do restante
        X_val_s, X_test_s, y_val_s, y_test_s = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42+i, shuffle=True)
        #FAZER TREINO
       
   