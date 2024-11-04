import os.path
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import alexandergovern
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
HIST_MAX = 6
ALPHA=0.05
cm=1/2.54


def get_sheet(sheet_id, range_name):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
        """
    
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    try:
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=sheet_id, range=range_name)
            .execute()
        )
        values = result.get("values", [])
        if not values:
            print("No data found.")
            return None
        
        return values
    except HttpError as err:
        print(err)
        return None

def gen_plot(data, bins, colors, labels, hatches, title, y_label, size):
    print(f"{title}, {y_label}, {labels}, {colors}, {hatches}, {size}")
    plt.figure(figsize=size)
    n, bins, patches = plt.hist(data, bins=bins, histtype='bar', color=colors, label=labels, edgecolor='black')
    plt.ylabel(y_label)
    plt.ylim(0, HIST_MAX)
    for patch_set, hatch in zip(patches, hatches):
        for patch in patch_set.patches:
            patch.set_hatch(hatch)
    plt.legend(labels=labels, loc='upper right')
    plt.savefig(os.path.join("figures", f"{title}.png"), bbox_inches='tight')
    #plt.show()
    plt.close()

def stat_analysis(data, labels):
    anova = f_oneway(*data)
    print(f"anova p-val: {anova.pvalue:.4f}, f-stat: {anova.statistic:.4f}")
    krusk = kruskal(*data)
    print(f"krusk p-val: {krusk.pvalue:.4f}, f-stat: {krusk.statistic:.4f}")
    alex = alexandergovern(*data)
    print(f" alex p-val: {alex.pvalue:.4f}, f-stat: {alex.statistic:.4f}")
    if len(data)>2:
        if (anova.pvalue < ALPHA) or (krusk.pvalue < ALPHA) or (alex.pvalue < ALPHA):
            for i in range(len(data)):
                for j in range(i+1,len(data)):
                    print(f"Compare {labels[i]}, {labels[j]}: {f_oneway(data[i], data[j])}")

if __name__ == "__main__":  
    order = {
        "first":"",
        "second":"///"
    }
    type = {
        "real":"purple",
        "vr":"orange"
    }
    two_row_three_col = (3.5,2)
    two_row_one_col = (3.5,2)

    # The ID and range of a sample spreadsheet.
    vr_real_id = "1eNymb-COFtsTHa_AlowTDaXOpbueqklVZNjURuJAaPo"
    real_vr_id = "1zbPZOyVYfKLXhKHUc6sw868l52LtRtBX7sjupGlpPi4"
    
    # realism
    # immersion
    simulations_realism_immersion_vr_first_range =  "Form Responses 1!G:H"
    simulations_realism_immersion_vr_second_range =  "Form Responses 1!W:X"
    simulations_realism_immersion_vr_first = np.asarray(get_sheet(vr_real_id, simulations_realism_immersion_vr_first_range))
    simulations_realism_immersion_vr_second = np.asarray(get_sheet(real_vr_id, simulations_realism_immersion_vr_second_range))
    titles = ["Immersiveness", "Realism"]
    colors = [type["vr"], type["vr"]]
    hatches = [order["first"], order["second"]]
    labels = ["VR First", "VR Second"]
    size = two_row_one_col
    bins = np.arange(1, 5 + 1.5) - 0.5
    for i in range(2): 
        print("====================================")
        col = simulations_realism_immersion_vr_first[0][i]
        print(col)
        title = titles[i]
        vr_first = simulations_realism_immersion_vr_first[1:,i].astype(int)
        vr_second = simulations_realism_immersion_vr_second[1:,i].astype(int)
        print(f"vr first  : {vr_first}")
        print(f"real first: {vr_second}")
        print(f"vr first  : mean: {np.mean(vr_first):.4f}, stddev: {np.std(vr_first):.4f}")
        print(f"real first: mean: {np.mean(vr_second):.4f}, stddev: {np.std(vr_second):.4f}")
        data = (vr_first,vr_second)
        stat_analysis(data, labels)
        gen_plot(data, bins=bins,colors=colors,labels=labels,hatches=hatches, title=title, y_label="Responses",size=size)

    #NASA TLX
    # Mental
    # Pysicsal
    # Tempporal
    # Performance
    # Effort
    # Frustration
    tlx_vr_real_vr_range =    "Form Responses 1!N:S"
    tlx_vr_real_real_range =  "Form Responses 1!AD:AI"
    tlx_values_vr_first    = np.asarray(get_sheet(vr_real_id, tlx_vr_real_vr_range))
    tlx_values_real_second = np.asarray(get_sheet(vr_real_id, tlx_vr_real_real_range))
    tlx_real_vr_vr_range =    "Form Responses 1!F:K"
    tlx_real_vr_real_range =  "Form Responses 1!AD:AI"
    tlx_values_real_first = np.asarray(get_sheet(real_vr_id, tlx_real_vr_vr_range))
    tlx_values_vr_second  = np.asarray(get_sheet(real_vr_id, tlx_real_vr_real_range))
    bins = np.arange(1, 5 + 1.5) - 0.5
    labels = ["VR First", "VR Second", "Real First", "Real Second"]
    colors = [type["vr"], type["vr"], type["real"], type["real"]]
    hatches = [order["first"], order["second"],order["first"], order["second"]]
    size = two_row_three_col
    for i in range(6):
        print("====================================")
        col = tlx_values_vr_first[0][i]
        text = col.split(' - ')
        print(f"{text[0]} : {text[1]}")
        title = text[0]
        caption = '\n'.join(wrap(text[1], 27)) 
        vr_first = tlx_values_vr_first[1:,i].astype(int)
        vr_second = tlx_values_vr_second[1:,i].astype(int)
        real_first = tlx_values_real_first[1:,i].astype(int)
        real_second = tlx_values_real_second[1:,i].astype(int)
        print(f"vr first   : {vr_first}")
        print(f"vr second  : {vr_second}")
        print(f"real first : {real_first}")
        print(f"real second: {real_second}")
        print(f"vr first   : mean: {np.mean(vr_first):.4f}, stddev: {np.std(vr_first):.4f}")
        print(f"vr second  : mean: {np.mean(vr_second):.4f}, stddev: {np.std(vr_second):.4f}")
        print(f"real first : mean: {np.mean(real_first):.4f}, stddev: {np.std(real_first):.4f}")
        print(f"real second: mean: {np.mean(real_second):.4f}, stddev: {np.std(real_second):.4f}")
        data = (vr_first, vr_second, real_first, real_second)
        stat_analysis(data, labels)
        gen_plot((vr_first, vr_second, real_first, real_second), bins=bins,colors=colors,labels=labels,hatches=hatches, title=title, y_label="Responses", size=size)
    labels = ["VR", "Real"]
    colors = [type["vr"], type["real"]]
    hatches = ["", ""]
    size = two_row_three_col
    for i in range(6):
        print("====================================")
        col = tlx_values_vr_first[0][i]
        text = col.split(' - ')
        title = text[0]+"_combine"
        print(f"{title} : {text[1]}")
        caption = '\n'.join(wrap(text[1], 27)) 
        vr_first = tlx_values_vr_first[1:,i].astype(int)
        vr_second = tlx_values_vr_second[1:,i].astype(int)
        real_first = tlx_values_real_first[1:,i].astype(int)
        real_second = tlx_values_real_second[1:,i].astype(int)
        vr = np.concatenate((vr_first, vr_second))
        real = np.concatenate((real_first, real_second))
        print(f"vr  : {vr}")
        print(f"real: {real}")
        print(f"vr  : mean: {np.mean(vr):.4f}, stddev: {np.std(vr):.4f}")
        print(f"real: mean: {np.mean(real):.4f}, stddev: {np.std(real):.4f}")
        data = (vr,real)
        stat_analysis(data, labels)
        gen_plot(data, bins=bins,colors=colors,labels=labels,hatches=hatches, title=title, y_label="Responses", size=size)

    # comfort
    # natrual
    # intuitive
    # pleasent
    # easy to use
    # inconsistancy
    inter_vr_real_vr_range =    "Form Responses 1!T:Z"
    inter_vr_real_real_range =  "Form Responses 1!AJ:AP"
    interaction_values_vr_first =  np.asarray(get_sheet(vr_real_id, inter_vr_real_vr_range))
    interaction_values_real_second =  np.asarray(get_sheet(vr_real_id, inter_vr_real_real_range))
    inter_real_vr_real_range =  "Form Responses 1!L:R"
    inter_real_vr_vr_range =    "Form Responses 1!AJ:AP"
    interaction_values_real_first =  np.asarray(get_sheet(real_vr_id, inter_real_vr_real_range))
    interaction_values_vr_second =  np.asarray(get_sheet(real_vr_id, inter_real_vr_vr_range))
    indxs = [0,3,4,5,6]
    titles = ["comfortable", "intuitive", "pleasant", "ease_of_use", "inconsistency"]
    labels = ["VR First", "VR Second", "Real First", "Real Second"]
    colors = [type["vr"], type["vr"], type["real"], type["real"]]
    hatches = [order["first"], order["second"],order["first"], order["second"]]
    bins = np.arange(1, 5 + 1.5) - 0.5
    size = two_row_three_col
    for j in range(5):
        i = indxs[j]
        print("====================================")
        #print(f"j:{j}, i:{i}, ")
        col = interaction_values_vr_first[0][i]
        caption = col
        title = titles[j]
        print(f"{title} : {caption}")
        #caption = '\n'.join(wrap(text[1], 27)) 
        vr_first = interaction_values_vr_first[1:,i].astype(int)
        vr_second = interaction_values_real_second[1:,i].astype(int)
        real_first = interaction_values_real_first[1:,i].astype(int)
        real_second = interaction_values_vr_second[1:,i].astype(int)
        print(f"vr first   : {vr_first}")
        print(f"vr second  : {vr_second}")
        print(f"real first : {real_first}")
        print(f"real second: {real_second}")
        print(f"vr first   : mean: {np.mean(vr_first):.4f}, stddev: {np.std(vr_first):.4f}")
        print(f"vr second  : mean: {np.mean(vr_second):.4f}, stddev: {np.std(vr_second):.4f}")
        print(f"real first : mean: {np.mean(real_first):.4f}, stddev: {np.std(real_first):.4f}")
        print(f"real second: mean: {np.mean(real_second):.4f}, stddev: {np.std(real_second):.4f}")
        data = (vr_first, vr_second, real_first, real_second)
        stat_analysis(data, labels)
        gen_plot((vr_first, vr_second, real_first, real_second), bins=bins,colors=colors,labels=labels,hatches=hatches, title=title, y_label="Responses", size=size)
    labels = ["VR", "Real"]
    colors = [type["vr"], type["real"]]
    hatches = ["", ""]
    size = two_row_three_col
    for j in range(5):
        i = indxs[j]
        print("====================================")
        col = interaction_values_vr_first[0][i]
        caption = col
        title = titles[j]+"_combine"
        print(f"{title} : {caption}")
        caption = '\n'.join(wrap(text[1], 27)) 
        vr_first = interaction_values_vr_first[1:,i].astype(int)
        vr_second = interaction_values_real_second[1:,i].astype(int)
        real_first = interaction_values_real_first[1:,i].astype(int)
        real_second = interaction_values_vr_second[1:,i].astype(int)
        vr = np.concatenate((vr_first, vr_second))
        real = np.concatenate((real_first, real_second))
        print(f"vr  : {vr}")
        print(f"real: {real}")
        print(f"vr  : mean: {np.mean(vr):.4f}, stddev: {np.std(vr):.4f}")
        print(f"real: mean: {np.mean(real):.4f}, stddev: {np.std(real):.4f}")
        data = (vr,real)
        stat_analysis(data, labels)
        gen_plot(data, bins=bins,colors=colors,labels=labels,hatches=hatches, title=title, y_label="Responses", size=size)


    # Comparison
    vr_first_compare_range =    "Form Responses 1!AQ:AQ"
    vr_first_compare_val =  np.asarray(get_sheet(vr_real_id, vr_first_compare_range))
    real_first_compare_range =  "Form Responses 1!AQ:AQ"
    real_first_compare_val =  np.asarray(get_sheet(real_vr_id, real_first_compare_range))
    bins = np.arange(1, 5 + 1.5) - 0.5
    labels = ["VR First", "Real First"]
    colors = [type["vr"], type["real"]]
    hatches = ["",""]
    size = two_row_one_col
    for i in range(1):
        print("====================================")
        col = vr_first_compare_val[0][i]
        caption = col
        title = "VR_real_similartity"
        print(f"{title} : {caption}")
        #caption = '\n'.join(wrap(text[1], 27)) 
        vr_first = vr_first_compare_val[1:,i].astype(int)
        vr_second = real_first_compare_val[1:,i].astype(int)
        print(f"vr first  : {vr_first}")
        print(f"real first: {vr_second}")
        print(f"vr first  : mean: {np.mean(vr_first):.4f}, stddev: {np.std(vr_first):.4f}")
        print(f"real first: mean: {np.mean(vr_second):.4f}, stddev: {np.std(vr_second):.4f}")
        data = (vr_first,vr_second)
        stat_analysis(data, labels)
        gen_plot(data, bins=bins,colors=colors,labels=labels,hatches=hatches, title=title, y_label="Responses", size=size)