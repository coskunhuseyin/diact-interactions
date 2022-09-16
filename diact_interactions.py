#!/usr/bin/env python
# coding: utf-8

# # The impact analysis of the US economy

# This code implements the impact analysis for the US economy through the system decomposition theory as introduced by H. Coskun, "Direct and indirect transactions and requirements," 2019 (<a href="https://osf.io/w2a4d" target="_blank">preprint</a>)

# The necessary packages for the code:
from asyncio import exceptions
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from numpy.linalg import inv
import requests
import zipfile
import io, os
import streamlit as st
import altair as alt
import urllib

try:

    # The input-output data published by the Bureau of Economic Analysis of the US can be downloaded as follows:
    @st.cache() 
    def get_IO_data():

        # Defining the zip file URL
        url = "https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesIO.zip"
        # Downloading the file by sending request to the URL
        response = requests.get(url)
        filestream = io.BytesIO(response.content)
        zipped = zipfile.ZipFile(filestream)

        # The use and make tables for the years from 1997 to 2020 can be loaded to the workspace form the downloaded data set. 
        # The use and make tables for 2020 can be selected and reshaped for subsequent computations. 
        # The 5th rows will be used for column labels, and the 0th columns will be used for row labels.
        strsm = ["Make", "Before", "PRO", "Sector"]
        strsu = ["Use", "Before", "PRO", "Sector"]
        with zipped as zip_file:
            for member in zip_file.namelist():
                if all(x in member for x in strsu):
                    zip_file.extract(member)   # does not work without extracting 
                    dfu_excel = pd.ExcelFile(member)
                    member_u = member
                elif all(x in member for x in strsm):
                    zip_file.extract(member)    # does not work without extracting 
                    dfm_excel = pd.ExcelFile(member)
                    member_m = member

        member_um = [member_u, member_m]    
        
        dfu_all = {}
        dfm_all = {}    
        sheets = []
        for sheetname in dfu_excel.sheet_names:
            if sheetname.isnumeric():
                sheets.append(sheetname)
                dfu_all[sheetname] = pd.read_excel(dfu_excel, sheet_name = sheetname, header = 5, index_col = 0)
        year_mx = [min(sheets), max(sheets)]
        df_codes = dfu_all[sheetname].index[1:16] 
        df_codes_names = dfu_all[sheetname].index[1:16] + ' -- ' + dfu_all[sheetname].iloc[1:16,0]

        for sheetname in dfm_excel.sheet_names:
            if sheetname.isnumeric():
                dfm_all[sheetname] = pd.read_excel(dfm_excel, sheet_name = sheetname, header = 5, index_col = 0)

        return dfu_all, dfm_all, df_codes, df_codes_names, year_mx, sheets, member_um 

except urllib.error.URLError as e:
    st.error(
        """
           **This app requires internet access.**
           Connection error: %s
        """
        % e.reason
    )

@st.cache 
def convert_df(df):
    return df.to_csv().encode('utf-8')

def displayRT(M, output_df, key_val):
    '''displays dataframes and enables download'''

    M_df = pd.DataFrame(M, index = df_codes, columns = df_codes)
    if rt == 'r':
        st.markdown(f'The {sc} {diact} demand distribution induced throughout the entire US economy in the year {year} \
            by the unit {inputlevelname} vector:') 
    elif rt == 't':     
        st.markdown(f'The {sc} {diact} demand distribution induced throughout the entire US economy in the year {year} \
            by the arbitrarily given {inputlevelname} vector y:')

    # for some reason, coloring does not work with streamlit 
    posis = df_codes.get_indexer(ts)
    def style_specific_cell(x):
        color = 'color:white;background-color:darkblue'
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        df.iloc[posis[0], posis[1]] = color
        df.iloc[posis[1], posis[0]] = color
        return df
    M_df.style.apply(style_specific_cell, axis=None)
    st.write(M_df)     

    M_csv = convert_df(M_df)
    st.download_button(
        label = "Download matrix as CSV",
        data = M_csv,
        file_name = key_val+'_'+sc+'_'+diact+'.csv', 
        mime = 'text/csv',
        key = key_val+'_'+sc+'_'+diact+'_matrix') 

    if rt == "r":
        o_temp_12 = output_df.loc[ts[1],ts[2]]
        o_temp_02 = output_df.loc[ts[0],ts[2]]
        output_df.loc[ts[1],ts[2]] = 1
        output_df.loc[ts[0],ts[2]] = 1

    try:
        if diact == "transfer": 
            st.write(f'**Interpretation of the matrix entries:** As an example, the results indicate that \
                the {diact} (total) demand from sector {ts[0]} directly and indirectly by sector {ts[1]} induced by \
                {ts[2]}(sector {ts[1]}) = \${output_df.loc[ts[1],ts[2]]:.2f} million is \${M_df[ts[1]][ts[0]]:.4f} million. In other words, \
                sector {ts[1]} needs \${M_df[ts[1]][ts[0]]:.4f} million worth of {diact} (total) input from sector {ts[0]} to meet \
                the {inputlevelname} of \${output_df.loc[ts[1],ts[2]]:.2f} million from itself. Similarly, the {diact} (total) demand \
                from sector {ts[1]} directly and indirectly by sector {ts[0]} induced by \
                {ts[2]}(sector {ts[0]}) = \${output_df.loc[ts[0],ts[2]]:.2f} million is \${M_df[ts[0]][ts[1]]:.4f} million. \
                In other words, sector {ts[0]} needs \${M_df[ts[0]][ts[1]]:.4f} million worth of {diact} (total) input from \
                {ts[1]} to meet the {inputlevelname} of \${output_df.loc[ts[0],ts[2]]:.2f} million from itself.') 
        else: 
            st.write(f'**Interpretation of the matrix entries:** As an example, the results indicate that \
                the total demand from sector {ts[0]} {diact}ly through other sectors by sector {ts[1]} induced by \
                {ts[2]}(sector {ts[1]}) = \${output_df.loc[ts[1],ts[2]]:.2f} million is \${M_df[ts[1]][ts[0]]:.4f} million. In other words, \
                sector {ts[1]} needs \${M_df[ts[1]][ts[0]]:.4f} million worth of {diact} input from sector {ts[0]} to meet \
                the {inputlevelname} of \${output_df.loc[ts[1],ts[2]]:.2f} million from itself. Similarly, the total demand \
                from sector {ts[1]} {diact}ly through other sectors by sector {ts[0]} induced by \
                {ts[2]}(sector {ts[0]}) = \${output_df.loc[ts[0],ts[2]]:.2f} million is \${M_df[ts[0]][ts[1]]:.4f} million. \
                In other words, sector {ts[0]} needs \${M_df[ts[0]][ts[1]]:.4f} million worth of {diact} input from \
                {ts[1]} to meet the {inputlevelname} of \${output_df.loc[ts[0],ts[2]]:.2f} million from itself.') 
    except:
        st.error('Please fill in the required information on the left panel.')

    if rt == "r":
        output_df.loc[ts[1],ts[2]] = o_temp_12
        output_df.loc[ts[0],ts[2]] = o_temp_02

def displayUM(M_df, key_val):
    '''displays dataframes and enables download'''

    st.write(M_df) 
    M_csv = convert_df(M_df)
    st.download_button(
        label = "Download matrix as CSV",
        data = M_csv,
        file_name = key_val+".csv",
        mime = 'text/csv',
        key = key_val+"_matrix") 

    try:
        if key_val == "U":
            st.write(f'**Interpretation of the matrix entries:** As an example, the results indicate that \
                the monetary value of the primary product of sector {ts[0]} required by sector {ts[1]} \
                is \${float(M_df[ts[1]][ts[0]]):.0f} million in the year {year}.')
        elif key_val == "M":
            st.write(f'**Interpretation of the matrix entries:** As an example, the results indicate that \
                the monetary value of the primary product of sector {ts[1]} produced by sector {ts[0]} \
                is \${float(M_df[ts[1]][ts[0]]):.0f} million in the year {year}.')
        elif key_val == "A":
            st.write(f'**Interpretation of the matrix entries:** As an example, the results indicate that \
                the monetary value of the primary product of sector {ts[0]} directly required by sector {ts[1]} \
                per unit gross demand from itself is \${M_df[ts[1]][ts[0]]:.4f} million in the year {year}.')
        elif key_val == "N":    
            st.write(f'**Interpretation of the matrix entries:** As an example, the results indicate that \
                the total monetary value of the primary product of sector {ts[0]} required by sector {ts[1]} \
                per unit final demand from itself is \${M_df[ts[1]][ts[0]]:.4f} million in the year {year}. \
                The cumulative requirements matrix is called the total requirements matrix in the input-output literature.')
    except:
        st.error("""**Please fill in the required information on the left panel.** """)

@st.cache
def dist_matrices():
    dfu_disp = dict()
    dfm_disp = dict()
    U = dict()
    M = dict()
    tau_c = dict()
    tau_s = dict()
    D = dict()
    B = dict()
    A = dict()
    A_df = dict()    
    N = dict()
    N_df = dict()    
    Ndiag = dict()
    Nd = dict()
    Ni = dict()
    Na = dict()
    Nc = dict() 
    Nt = dict()
    for years in sheets:

        dfu_year = dfu_all[years][:24]
        dfm_year = dfm_all[years][:17]

        dfu_disp[years] = dfu_year.replace('...',0)
        dfm_disp[years] = dfm_year.replace('...',0)

        # The use and make matrices can be extracted from the use dataframe as a numeric array after replacing "..." in the table entries by 0 as follows:
        U[years] = dfu_year.iloc[1:18,1:-9].replace('...',0).to_numpy()
        M[years] = dfm_year.iloc[1:16,1:-1].replace('...',0).to_numpy()

        # The prodcut and sector outputs ($\tau_c$ and $\tau_s$) can be computed as follows:
        tau_c[years] = np.sum(M[years], axis=0)
        tau_s[years] = np.sum(M[years], axis=1)

        # The scaled use and make matrices ($D$ and $B$) becomes:
        D[years] = np.matmul(M[years],inv(np.diag(tau_c[years])))
        B[years] = np.matmul(U[years],inv(np.diag(tau_s[years])))

        # The industry-by-industry direct requirements matrix, $A$, is: 
        A[years] = np.matmul(D[years],B[years])
        A_df[years] = pd.DataFrame(A[years], index = df_codes, columns = df_codes)

        # The corresponding industry-by-industry cumulative requirements matrix, $N$, becomes: 
        N[years] = inv(np.identity(15)-A[years])
        N_df[years] = pd.DataFrame(N[years], index = df_codes, columns = df_codes)
        # The diagonal matrix whose diagonal elements are the same as the elements of $N$, $Ndiag$, can be computed as follows:
        Ndiag[years] = np.diag(N[years].diagonal()) 

        if sc == "simple":
            Nd[years] = np.matmul(A[years],Ndiag[years])
            Ni[years] = np.matmul(A[years],N[years]-Ndiag[years])
            Na[years] = np.matmul(inv(Ndiag[years]),N[years]-Ndiag[years])
            Nc[years] = np.matmul(np.identity(15)-inv(Ndiag[years]),N[years])
            Nt[years] = np.matmul(A[years],N[years])    # Notice that Nt = Nd + Ni, that is np.max(Nt-(Nd+Ni))=0            
        if sc == "composite":
            NinvNd = np.matmul(N[years],inv(Ndiag[years]))
            Nd[years] = A[years]
            Ni[years] = np.matmul(A[years],NinvNd-np.identity(15))
            Na[years] = np.matmul(inv(Ndiag[years]),NinvNd-np.identity(15))
            Nc[years] = np.matmul(np.identity(15)-inv(Ndiag[years]),NinvNd)
            Nt[years] = np.matmul(A[years],NinvNd)    # Notice that Nt = Nd + Ni, that is np.max(Nt-(Nd+Ni))=0            

    return dfu_disp, dfm_disp, A_df, N_df, Nd, Ni, Na, Nc, Nt 

# the data and related information are called here 
dfu_all, dfm_all, df_codes, df_codes_names, year_mx, sheets, member_um = get_IO_data() 

with st.sidebar:
    
    user_inputs_section = st.container()

    with user_inputs_section:
        st.title("User inputs")        
        st.write("Please provide your input for the following queries for the analysis of the US economy. \
            The analysis will be presented on the page for the given user inputs on this panel.")
        
        # The year of analysis
        st.subheader("Year of analysis")
        year = st.text_input(f"Choose the year from {year_mx[0]} to {year_mx[1]} for which the numerical results will be presented.", value = year_mx[1]) 

        st.subheader("Decomposition level of interactions")
        # An arbitrary final or cumulative demand vector, $y$ or $\tau$, need to be requested from the user.

        try:
            sc_set = ["simple", "composite"] 
            sc_sel = st.multiselect(
                "Choose the decomposition level of the transactions.",
                sc_set, default = sc_set[0]) 
            sc = sc_sel[0] 

            if sc == "simple":
                inputlevelname = "final demands" 
                try: 
                    external_output = st.text_input(f"Enter {inputlevelname} for each of the 15 sectors separated by a comma. \
                        The IOCodes of sectors given at the bottom of this panel below.", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15") 
                    y = list(map(float, external_output.strip().split(",")))[:15] 
                    Y_df = pd.DataFrame(y, index = df_codes, columns = ["y"])
                    st.write(f"The {inputlevelname} vector in million dollars you entered in tabular form is:", Y_df.T.style.format("{:.2f}")) 
                    # For the analysis of pairwise interactions, the system decomposition theory converts the final demands vector to a diagonal matrix:
                    Y = np.diag(y)
                    output = Y
                    output_df = Y_df
                except:
                    st.error(f"""**Please enter {inputlevelname} for each sector separated by comma.** """)

            if sc == "composite":
                inputlevelname = "gross demands"
                try:
                    total_output = st.text_input(f"Enter {inputlevelname} for each of the 15 sectors separated by a comma. \
                        The IOCodes of sectors given at the bottom of this panel below.", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15") 
                    tau_arb = list(map(float, total_output.strip().split(",")))[:15] 
                    Tau_df = pd.DataFrame(tau_arb, index = df_codes, columns = ["tau"]) 
                    st.write(f"The {inputlevelname} vector in million dollars you entered in tabular form is:", Tau_df.T.style.format("{:.2f}")) 
                    # For the analysis of pairwise interactions, the system decomposition theory converts the cumulative demands vector to a diagonal matrix:
                    Tau = np.diag(tau_arb)
                    output = Tau
                    output_df = Tau_df
                except:
                    st.error(f"""**Please enter {inputlevelname} for each sector separated by comma.** """)
        except:
            st.error("""**Please choose one of the decomposition levels from the dropdown menu.** """) 

        st.subheader("Type of interactions")
        # The user input for the direct, indirect, acyclic, cycling, and transfer (total) demand distributions: 
        try:
            diact_set = ["direct", "indirect", "acyclic", "cycling", "transfer"] 
            diact_sel = st.multiselect(
                "Choose the nature of interactions need to be quantified in the US economy.",
                diact_set, default = diact_set[1])
            diact = diact_sel[0] 
        except:
            st.error("""**Please choose one of the interaction types from the dropdown menu.** """)

        st.subheader("Pairwise interactions")
        try:
            two_sectors = st.multiselect(    
                "Choose two sectors whose pairwise simple relationships need to be quantified.",
                df_codes_names, default = [df_codes_names[0], df_codes_names[4]])
            two_sectors = two_sectors[:2]
            ts = [two_sectors[0].split()[0]]
            ts.append(two_sectors[1].split()[0])
        except:
            st.error("""**Please choose two sectors from the dropdown menu.** """)

        try:    
            if sc == "simple":
                ts.append('y')
            elif sc == "composite":
                ts.append('tau')
        except:
            st.error("""**Please fill in the required information on the left panel.** """)    

        st.write("As you change these options, the results on the right hand panel will automatically be updated accordingly.") 
        st.write(" ") 
        st.write(" ")
        st.write(" ")                

headertitle = st.container()
data = st.container()
results = st.container() 
bibliography = st.container()

with headertitle:
    st.title("The diact economic interactions")
    st.write("This interactive application is designed for the quantification of the pairwise \
        direct, indirect, acyclic, cycling, and transfer (diact) economic interactions between any two sectors \
        in the US economy. The app is based on the **system decomposition theory**, \
        which is briefly discussed at the bottom of the page.") 

try:

    with data:
        st.subheader("Data and standard results") 
        st.write(f"The input-output data used in the app is annually published by the Bureau of Economic Analysis of the US \
            ([BEA](https://apps.bea.gov/iTable/iTable.cfm?isuri=1&reqid=151&step=1)). \
            The reshaped use and make matrices as well as the standard direct and total requirements matrices are presented \
            in the tabs below for the year {year} and interpreted in the context of the system decomposition theory. \
            The names and the IOCodes of the sectors in this 15-sector aggregated system are listed in the first two columns of the use and make matrices.") 

        dfu_disp, dfm_disp, A_df, N_df, Nd, Ni, Na, Nc, Nt = dist_matrices() 

        use_t, make_t, sdirect_t, cumulative_t = st.tabs(["Use matrix", "Make matrix", "Direct requirements matrix", "Cumulative requirements matrix"])

        # the use and make matrices in two columns page layout
        with use_t:
            displayUM(dfu_disp[year], "U")
        with make_t:
            displayUM(dfm_disp[year], "M")
        with sdirect_t:
            displayUM(A_df[year], "A")
        with cumulative_t:
            displayUM(N_df[year], "N")     

    Nd_df = dict()
    Ni_df = dict()
    Na_df = dict()
    Nc_df = dict()
    Nt_df = dict()
    Td = dict()
    Ti = dict()
    Ta = dict()
    Tc = dict()
    Tt = dict() 
    Td_df = dict()
    Ti_df = dict()
    Ta_df = dict()
    Tc_df = dict()
    Tt_df = dict()
    for years in sheets:
        Nd_df[years] = pd.DataFrame(Nd[years], index = df_codes, columns = df_codes)
        Ni_df[years] = pd.DataFrame(Ni[years], index = df_codes, columns = df_codes)    
        Na_df[years] = pd.DataFrame(Na[years], index = df_codes, columns = df_codes)
        Nc_df[years] = pd.DataFrame(Nc[years], index = df_codes, columns = df_codes)    
        Nt_df[years] = pd.DataFrame(Nt[years], index = df_codes, columns = df_codes)    

        Td[years] = np.matmul(Nd[years],output)
        Ti[years] = np.matmul(Ni[years],output)
        Ta[years] = np.matmul(Na[years],output)
        Tc[years] = np.matmul(Nc[years],output)
        Tt[years] = np.matmul(Nt[years],output)    # Notice that Tt = Td + Ti, that is np.max(Tt-(Td+Ti))=0 

        Td_df[years] = pd.DataFrame(Td[years], index = df_codes, columns = df_codes)
        Ti_df[years] = pd.DataFrame(Ti[years], index = df_codes, columns = df_codes)
        Ta_df[years] = pd.DataFrame(Ta[years], index = df_codes, columns = df_codes)
        Tc_df[years] = pd.DataFrame(Tc[years], index = df_codes, columns = df_codes)
        Tt_df[years] = pd.DataFrame(Tt[years], index = df_codes, columns = df_codes)

    Td_df_ts = dict()
    Ti_df_ts = dict()
    Ta_df_ts = dict()
    Tc_df_ts = dict()
    Tt_df_ts = dict()
    try: 
        for years in sheets:
            Td_df_ts[years] = Td_df[years].loc[ts[0],ts[1]]
            Ti_df_ts[years] = Ti_df[years].loc[ts[0],ts[1]]
            Ta_df_ts[years] = Ta_df[years].loc[ts[0],ts[1]]
            Tc_df_ts[years] = Tc_df[years].loc[ts[0],ts[1]]        
            Tt_df_ts[years] = Tt_df[years].loc[ts[0],ts[1]]
    except:
        st.error("""**Please fill in the required information on the left panel.** """)

    with results:
        st.subheader("The requirements and transactions matrices")
        st.write(f"The system decomposition theory enables the quantification of the pairwise composite and simple \
            diact interactions between any two sectors of the system for the first time in literature. The diact requirements and transactions \
            matrices induced by arbitrarily given final or gross demand vectors are presented in this section. \
            These transactions matrices represent the demand distributions throughout the system \
            induced by the given final or gross demands. The pairwise diact demand interactions between the two given \
            sectors from the year {year_mx[0]} to {year_mx[1]} are also graphically presented. Moreover, \
            the total {sc} demands from and by these two sectors for the year {year} are depicted below as well.")

        try:        
            graphs_t, distributions_t, transactions_t = st.tabs(["Graphical representations", "Requirements matrix", "Transactions matrix"])

            with graphs_t: 
                # The pairwise interactions are visualized in this section. 

                if len(ts)<2:
                    st.error("""**Please fill in the required information on the left panel.** """)
                elif len(sc)==0:
                    st.error("""**Please fill in the required information on the left panel.** """)

                col1, col2 = st.columns(2)

                with col1:
                    st.caption(f"**Dynamics of the given pairwise {sc} {diact} interaction \
                        induced by the given {inputlevelname}.**")

                    def pairwise_dit(T_ts):
                        T_ts_df = pd.DataFrame.from_dict(T_ts, orient='index', columns=[f"{diact} demand"])
                        T_ts_df.insert(0, "year", sheets, True)
                        return T_ts_df                    
                    
                    title_pdit=f"{sc} {diact} demand from sector {ts[0]} by sector {ts[1]}"
                    if diact == "direct":
                        pairwise_df = pairwise_dit(Td_df_ts)
                    elif diact == "indirect":
                        pairwise_df = pairwise_dit(Ti_df_ts) 
                    if diact == "acyclic":
                        pairwise_df = pairwise_dit(Ta_df_ts)
                    elif diact == "cycling":
                        pairwise_df = pairwise_dit(Tc_df_ts) 
                    elif diact == "transfer":
                        pairwise_df = pairwise_dit(Tt_df_ts)

                    pairwise_chart = alt.Chart(pairwise_df).mark_line().encode(
                        x = alt.X('year', 
                            axis=alt.Axis(
                            grid=False,
                            values=['2000','2005','2010','2015','2020'])),
                        y = alt.Y(pairwise_df.columns[1], 
                            axis=alt.Axis(title=title_pdit)),
                        tooltip = ["year", pairwise_df.columns[1]]
                            ).properties(width=330, height=360)

                    st.altair_chart(pairwise_chart, use_container_width=False)

                with col2:
                    st.caption(f"**The {sc} {diact} demands by and from the given sectors in {year} \
                        induced by the given {inputlevelname}.**")

                    df_codes_df = pd.DataFrame(df_codes, index=df_codes, columns=["sectors"])
                    sects = pd.concat([df_codes_df, df_codes_df], axis=0)
                    from_series = pd.DataFrame(["from"]*15, index=df_codes, columns=["demand"])
                    by_series = pd.DataFrame(["by"]*15, index=df_codes, columns=["demand"])
                    def tau_dit(T_df):
                        tau_df = {}
                        tau_ts_df = {}
                        for years in sheets:
                            tau_df_out = pd.DataFrame(T_df[years].sum(axis=1), index = df_codes, columns = [f"total {diact} demand"])
                            tau_df_out = pd.concat([from_series, tau_df_out], axis=1)
                            tau_df_in = pd.DataFrame(T_df[years].sum(axis=0), index = df_codes, columns = [f"total {diact} demand"])
                            tau_df_in = pd.concat([by_series, tau_df_in], axis=1)
                            tau_final = pd.concat([tau_df_out, tau_df_in], axis=0)
                            tau_df[years] = pd.concat([sects,tau_final], axis=1)
                            tau_ts_df[years] = tau_df[years].loc[ts[:2]]
                        return tau_ts_df, tau_df 

                    title_dit=f"total {sc} {diact} demands by and from sectors {ts[0]} and {ts[1]}"                    
                    if diact == "direct":                    
                        tau_ts_df, tau_df = tau_dit(Td_df)                    
                    elif diact == "indirect":
                        tau_ts_df, tau_df = tau_dit(Ti_df)                    
                    if diact == "acyclic":
                        tau_ts_df, tau_df = tau_dit(Ta_df)                    
                    elif diact == "cycling":
                        tau_ts_df, tau_df = tau_dit(Tc_df)                    
                    elif diact == "transfer": 
                        tau_ts_df, tau_df = tau_dit(Tt_df)        

                    # stored variables into a pickle file for later use:
                    import pickle
                    file_name = f"{sc}_{diact}_{ts[0]}_{ts[1]}_transferred_data"
                    # Open the file for writing
                    with open(file_name,'wb') as my_file_obj:
                        pickle.dump([pairwise_df, tau_ts_df, Td_df, Ti_df, Ta_df, Tc_df, Tt_df, \
                            Nd_df, Ni_df, Na_df, Nc_df, Nt_df], my_file_obj)

                    tau_chart = alt.Chart(tau_ts_df[year]).mark_bar().encode(
                        alt.Column('sectors'), 
                        alt.X('demand'),
                        alt.Y(tau_ts_df[year].columns[2], title=title_dit, axis=alt.Axis(grid=False)),
                        alt.Color('demand'),
                        tooltip = [tau_ts_df[year].columns[2]]
                        ).properties(width=110, height=230)
    
                    st.altair_chart(tau_chart, use_container_width=False)

            with distributions_t: 
                # The direct, indirect, acyclic, cycling, transfer requirements induced by unit final demand vector, $Nd$, $Ni$, $Na$, $Nc$, and $Nt$, can be computed  
                # and these matrices can be expressed in dataframe format as follows:

                rt = "r"
                if diact == "direct":
                    displayRT(Nd[year], output_df, "Nd")
                elif diact == "indirect":
                    displayRT(Ni[year], output_df, "Ni")
                elif diact == "acyclic":
                    displayRT(Na[year], output_df, "Na")
                elif diact == "cycling":
                    displayRT(Nc[year], output_df, "Nc")
                elif diact == "transfer":
                    displayRT(Nt[year], output_df, "Nt")

            with transactions_t:
                # The direct, indirect, transfer demand distributions induced by given final demand vector $y$, $Td$, $Ti$, and $Tt$, can be computed 
                # and these matrices can be expressed in dataframe format as follows:

                rt = "t"
                if diact == "direct":
                    displayRT(Td[year], output_df, "Td")
                elif diact == "indirect":
                    displayRT(Ti[year], output_df, "Ti")
                elif diact == "acyclic":
                    displayRT(Ta[year], output_df, "Ta")
                elif diact == "cycling":
                    displayRT(Tc[year], output_df, "Tc")
                elif diact == "transfer":
                    displayRT(Tt[year], output_df, "Tt")

        except:
            st.error("""**Please fill in the required information on the left panel.** """)

except NameError:
        st.error("""**Please fill in the required information on the left panel.** """)

finally:
    
    with bibliography:
        st.subheader("References") 

        st.markdown("This interactive diact interactions application essentially implements the methods introduced in the following paper \
            for the analysis of economic systems based on the system decomposition theory:")
        st.write("* [Husna B. Coskun and Huseyin Coskun. Direct and indirect transactions and requirements, (2019)](https://osf.io/w2a4d/)")
        st.write("Additional novel formulations, such as the acyclic and cycling requirements and transactions \
            are also implemented. The system decomposition theory is also briefly outlined in the expander panel below.")

        with st.expander("System decomposition theory"):

            st.write("The system decomposition theory has recently been introduced for the holistic comprehensive analysis of \
                nonlinear dynamic compartmental systems. The theory solves multiple long-standing open problems \
                including the quantification of pairwise direct, indirect, cycling, acyclic, transfer (total) interactions within the system. It enables \
                tracking and back-tracing any arbitrary system flow and storage segment throughout the system, \
                as well as their impact, decomposition, evolution, dispersion, and contraction. The system decomposition theory \
                enhances the strength and extends the applicability of the state-of-the-art techniques and provides \
                significant advancements in theory, methodology, and practicality.")
            st.write("The system decomposition theory enables the analysis of \
                systems modelling real-world phenomena in multiple scientific fields. The theory \
                and associated methods may prove useful, as examples, in quantitative systems pharmacology (pharmacometrics, \
                drug development, ADME, PK/PD, drug metabolism and calibration, and exposure), biomedical and chemical systems \
                (molecular interactions, metabolic networks, pathway analysis, and model reduction), epidemiology \
                (population dynamics and infectious diseases, ecological systems (ecosystems ecology, food webs, microecology), \
                economic systems (supply, demand, production, consumption times, rates, and efficiencies, system responses, \
                elasticities, price and growth indices, business cycles, sustainability based on input-output data), and \
                data science (system structure, deep learning, and neural networks).")
            st.write("The system decomposition theory is introduced in a series of articles. The links below \
                are for the preprint versions of a selected set of these articles, but the peer-reviewed versions can be \
                found within the linked pages.") 

            st.markdown("* [Huseyin Coskun. Static ecological system analysis, Journal of Theoretical Ecology, (2020)](https://osf.io/zqxc5/)")
            st.markdown("* [Huseyin Coskun. Static ecological system measures, Journal of Theoretical Ecology, (2020)](https://osf.io/g4xzt/)")
            st.markdown("* [Huseyin Coskun. Dynamic ecological system analysis. Heliyon, 5 (2019)](https://osf.io/35xkb/)")
            st.markdown("* [Huseyin Coskun. Dynamic ecological system measures. Results in Applied Mathematics, (2019)](https://osf.io/j2pd3/)")
            st.markdown("* [Huseyin Coskun. Nonlinear decomposition principle and fundamental matrix solutions for dynamic compartmental systems. Discrete and Continuous Dynamical Systems -- Series B, 24 (2019) 6553-6605](https://osf.io/cyrzf/)")
    