import nltk
from nltk.tokenize import sent_tokenize
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import re
from collections import deque, Counter

#pobieram model tokenizacji, aby podzielic tekst na zdania (lub pare zdan)
nltk.download('punkt')
nltk.download('punkt_tab')

#pliki .txt zawierajace tresc sagi wiedzminskiej
filenames = [
    'Blood of Elves.txt', 'Times of Contempt.txt',
    'Baptism of Fire.txt', 'The Tower of the Swallow.txt', 'The Lady of the Lake.txt'
]

#mapa postaci zawierajaca klucze (postaci) i wartosci (aliasy, synonimy)
base_characters_map = {
    "Geralt": [
        "geralt of rivia", "geralt", "white wolf", "gwynbleidd",
        "butcher of blaviken", "ravix of fourhorn"
    ],
    "Yennefer": [
        "yennefer", "yen", "yenna", "horsewoman of war", "yennefer of vengerberg"
    ],
    "Ciri": [
        "ciri", "cirilla", "lion cub of cintra", "falka", "zireael",
        "swallow", "lady of space and time", "lady of the lake",
        "cirilla fiona elen riannon"
    ],
    "Triss": [
        "triss", "merigold", "triss merigold", "fourteenth of the hill", "fearless"
    ],
    "Dandelion": [
        "dandelion", "dandilion", "julian alfred pankratz",
        "viscount de lettenhove", "sandpiper", "poet", "bard"
    ],
    "Milva": ["milva", "maria barring", "kite", "sorrel"],
    "Regis": ["regis", "emiel regis", "emiel regis rohellec terzieff-godefroy", "vampire"],
    "Cahir": ["cahir", "cahir mawr dyffryn aep ceallach", "black knight", "nightmare of cintra"],
    "Angouleme": ["angouleme", "angoulême"],
    "Vesemir": ["vesemir"],
    "Lambert": ["lambert"],
    "Eskel": ["eskel"],
    "Coen": ["coen"],
    "Philippa Eilhart": ["philippa", "philippa eilhart", "she-wolf of the court"],
    "Francesca Findabair": ["francesca", "francesca findabair", "enid an gleanna", "daisy of the valleys"],
    "Fringilla Vigo": ["fringilla", "fringilla vigo"],
    "Keira Metz": ["keira", "keira metz"],
    "Margarita Laux-Antille": ["margarita", "margarita laux-antille", "rita"],
    "Sheala de Tancarville": ["sheala", "sile", "sheala de tancarville", "sile de tancarville"],
    "Sabrina Glevissig": ["sabrina", "sabrina glevissig", "daughter of the kaedwenian wilderness"],
    "Assire var Anahid": ["assire", "assire var anahid"],
    "Ida Emean aep Sivney": ["ida", "ida emean", "aen saevherne"],
    "Tissaia de Vries": ["tissaia", "tissaia de vries", "the archmistress"],
    "Lydia van Bredevoort": ["lydia", "lydia van bredevoort"],
    "Marti Sodergren": ["marti", "marti sodergren"],
    "Vilgefortz": ["vilgefortz", "vilgefortz of roggeveen"],
    "Emhyr": [
        "emhyr", "emhyr var emreis", "duny", "white flame",
        "white flame dancing on the barrows of his enemies",
        "urcheon of erlenwald", "deithwen addan yn carn aep morvudd"
    ],
    "Bonhart": ["bonhart", "leo bonhart"],
    "Stefan Skellen": ["skellen", "stefan skellen", "tawny owl", "coroner"],
    "Rience": ["rience"],
    "Schirru": ["schirru"],
    "Vattier de Rideaux": ["vattier", "vattier de rideaux", "chief of military intelligence"],
    "Mistle": ["mistle"],
    "Kayleigh": ["kayleigh"],
    "Giselher": ["giselher"],
    "Iskra": ["iskra"],
    "Reef": ["reef"],
    "Asse": ["asse"],
    "Foltest": ["foltest", "king foltest", "lord of temeria"],
    "Meve": ["meve", "queen meve", "the white queen"],
    "Henselt": ["henselt", "king henselt", "boar"],
    "Demavend": ["demavend", "demavend iii"],
    "Calanthe": ["calanthe", "lioness of cintra", "queen calanthe", "ard rhena"],
    "Pavetta": ["pavetta"],
    "Esterad Thyssen": ["esterad", "esterad thyssen", "king of kovir"],
    "Anna Henrietta": ["anna henrietta", "anarietta"],
    "Zoltan Chivay": ["zoltan", "zoltan chivay"],
    "Yarpen Zigrin": ["yarpen", "yarpen zigrin"],
    "Sigismund Dijkstra": ["dijkstra", "sigismund dijkstra", "count dijkstra", "sigi"],
    "Nenneke": ["nenneke", "mother nenneke"],
    "Shani": ["shani"],
    "Iola": ["iola"],
    "Jarre": ["jarre"],
    "Dudu": ["dudu", "tellico lunngrevink letorte", "biberveldt"],
    "Codringher": ["codringher"],
    "Fenn": ["fenn", "jacob fenn"],
    "Essi Daven": ["essi", "essi daven", "little eye"],
    "Borch Three Jackdaws": ["borch", "borch three jackdaws", "villentretenmerth", "gold dragon"],
    "Renfri": ["renfri", "shrike"],
    "Istredd": ["istredd"],
    "Stregobor": ["stregobor"],
    "Crach an Craite": ["crach", "crach an craite", "wild boar of the sea", "tirth ys muire"],
    "Mousesack": ["mousesack"],
    "Lara Dorren": ["lara", "lara dorren", "lara dorren aep shiadhal"],
    "Avallac'h": ["avallac'h", "crevan espane aep caomhan macha", "fox"],
    "Eredin": ["eredin", "eredin breacc glas", "king of the wild hunt", "sparrowhawk"],
    "Auberon Muircetach": ["auberon", "king of the alders", "auberon muircetach"],
    "Ihuarraquax": ["ihuarraquax", "little horse"],
    "Toruviel": ["toruviel"],
    "Yaevinn": ["yaevinn"],
    "Isengrim Faoiltiarna": ["isengrim", "isengrim faoiltiarna", "iron wolf"],
    "Filavandrel": ["filavandrel", "filavandrel aen fidhaill"]
}


def prepare_patterns(char_map):
    patterns = {}
    local_map = char_map.copy()

    for main_name, aliases in local_map.items():
        #sortujemy aliasy po długości, aby uniknąć problemów (najdłuższe dopasuj najpierw)
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_aliases]
        pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
        patterns[main_name] = re.compile(pattern_str)
    return patterns


def analyze_basic(sentences, patterns):
    G = nx.Graph()
    for sentence in sentences:
        found = set()
        s_lower = sentence.lower()
        for name, pat in patterns.items():
            if pat.search(s_lower):
                found.add(name)

        if len(found) > 1:
            for u, v in itertools.combinations(sorted(found), 2):
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    return G


import re
import itertools
import random
from collections import Counter
import networkx as nx


def analyze_context(sentences, char_map):
    patterns = prepare_patterns(char_map)

    # Mapa aliasów (bez zmian)
    ambiguous_map = {
        r'\bwitcher(:?s)?\b': ["Geralt", "Vesemir", "Lambert", "Eskel", "Coen"],
        r'\bsorceress(?:es)?\b': [
            "Yennefer", "Triss", "Philippa Eilhart", "Keira Metz", "Fringilla Vigo",
            "Margarita Laux-Antille", "Sheala de Tancarville", "Sabrina Glevissig",
            "Assire var Anahid", "Ida Emean aep Sivney", "Francesca Findabair"
        ],
        r'\bwitch?\b': ["Yennefer", "Triss", "Philippa Eilhart", "Keira Metz",
                        "Fringilla Vigo", "Margarita Laux-Antille", "Sheala de Tancarville",
                        "Sabrina Glevissig", "Assire var Anahid", "Ida Emean aep Sivney", "Francesca Findabair"
                        ],
        r'\bking(?:s)?\b': [
            "Foltest", "Henselt", "Demavend", "Esterad Thyssen",
            "Auberon Muircetach", "Eredin"
        ],
        r'\bqueen(?:s)?\b': ["Meve", "Calanthe", "Francesca Findabair"],
        r'\bprincess(?:es)?\b': ["Ciri", "Pavetta", "Anna Henrietta"],
        r'\belv(?:es|en)\b': [
            "Francesca Findabair", "Ida Emean aep Sivney", "Toruviel", "Yaevinn",
            "Isengrim Faoiltiarna", "Filavandrel", "Avallac'h", "Eredin",
            "Auberon Muircetach", "Iskra"
        ],
        r'\belf\b': [
            "Francesca Findabair", "Ida Emean aep Sivney", "Toruviel", "Yaevinn",
            "Isengrim Faoiltiarna", "Filavandrel", "Avallac'h", "Eredin",
            "Auberon Muircetach", "Iskra"
        ],
        r'\bdwar(?:ves|f)\b': ["Zoltan Chivay", "Yarpen Zigrin"],
        r'\bwizard?\b': ["Vilgefortz", "Stregobor", "Istredd", "Rience"],
        r'\bsorcerer?\b': ["Vilgefortz", "Stregobor", "Istredd", "Rience"],
        r'\bmagician?\b': ["Vilgefortz", "Stregobor", "Istredd", "Rience"],
        r'\brat?\b': ["Mistle", "Kayleigh", "Giselher", "Iskra", "Reef", "Asse"],
        r'\bscoia\'?tael\b': ["Isengrim Faoiltiarna", "Yaevinn", "Toruviel", "Filavandrel"],
        r'\bsquirrels?\b': ["Isengrim Faoiltiarna", "Yaevinn", "Toruviel", "Filavandrel"],
        r'\bsp(?:y|ies)\b': ["Sigismund Dijkstra", "Vattier de Rideaux", "Fenn", "Codringher"]
    }

    amb_patterns = {re.compile(k): v for k, v in ambiguous_map.items()}

    G = nx.Graph()

    # Iterujemy po indeksach, aby mieć dostęp do sąsiednich zdań
    n_sentences = len(sentences)

    for i, sentence in enumerate(sentences):
        found = set()
        s_lower = sentence.lower()

        # 1. Szukaj bezpośrednich dopasowań (imion) w OBECNYM zdaniu
        for name, pat in patterns.items():
            if pat.search(s_lower):
                found.add(name)

        # 2. Obsługa aliasów z analizą OKNA (p=5: 2 wstecz, obecne, 2 w przód)
        # Jeśli w obecnym zdaniu jest alias (np. 'king')...
        matches_alias = False
        active_candidates = set()  # Zbiór kandydatów do sprawdzenia

        for pat, candidates in amb_patterns.items():
            if pat.search(s_lower):
                matches_alias = True
                active_candidates.update(candidates)

        if matches_alias and active_candidates:
            # Definiujemy zakres okna
            start_idx = max(0, i - 5)
            end_idx = min(n_sentences, i + 6)  # slice jest 'exclusive', więc i+3 bierze do i+2 włącznie

            # Łączymy zdania z okna w jeden ciąg tekstowy dla łatwiejszego przeszukiwania
            window_text = " ".join(sentences[start_idx:end_idx]).lower()

            candidate_counts = []

            # Sprawdzamy wystąpienia konkretnych kandydatów w CAŁYM OKNIE
            for cand in active_candidates:
                if cand in patterns:
                    # Używamy findall, aby policzyć ile razy imię pada w oknie
                    count = len(patterns[cand].findall(window_text))
                    if count > 0:
                        candidate_counts.append((cand, count))

            # Logika wyboru: Max, a potem Random
            if candidate_counts:
                # Znajdź najwyższą liczbę wystąpień
                max_val = max(c[1] for c in candidate_counts)

                # Wybierz wszystkich, którzy mają ten max (remis)
                best_matches = [c[0] for c in candidate_counts if c[1] == max_val]

                # Losuj jednego z najlepszych
                chosen_one = random.choice(best_matches)
                found.add(chosen_one)

        # 3. Aktualizacja grafu
        found_list = list(found)
        if len(found_list) > 1:
            for u, v in itertools.combinations(sorted(found_list), 2):
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G

full_text = ""
for name in filenames:
    with open(name, 'r', encoding='utf-8') as f:
        full_text += " " + f.read()
sentences = sent_tokenize(full_text)

#bez aliasow dla wiecej niz 1 osoby
print("analiza podstawowa...")
patterns_basic = prepare_patterns(base_characters_map)
G1 = analyze_basic(sentences, patterns_basic)

#z aliasami dodatkowymi
print("analiza kontekstowa...")
G2 = analyze_context(sentences, base_characters_map)

#obliczanie roznicy
c1 = nx.degree_centrality(G1)
c2 = nx.degree_centrality(G2)
all_nodes = set(c1.keys()) | set(c2.keys())
c_diff = {node: c2.get(node, 0) - c1.get(node, 0) for node in all_nodes}
sorted_diff = sorted(c_diff.items(), key=lambda x: x[1])

edge_diff = {}
all_edges = set(G1.edges()) | set(G2.edges())
for u, v in all_edges:
    if u > v: u, v = v, u
    w1 = G1[u][v]['weight'] if G1.has_edge(u, v) else 0
    w2 = G2[u][v]['weight'] if G2.has_edge(u, v) else 0
    diff = w2 - w1
    if diff != 0:
        edge_diff[(u, v)] = diff

#top 10 centralnosci (dla analizy kontekstowej)
def centrality():
    print("\n--- Top 10 Centralność (Model Context) ---")
    sorted_centrality_g2 = sorted(c2.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, score in sorted_centrality_g2:
        print(f"{name}: {score:.3f}")

#wizualizacje
#kolory dla roznych grup
group_colors = {
    "Ciri": "#E0FFFF",
    "Witchers": "#D3D3D3",
    "Sorceresses": "#E6E6FA",
    "Rulers": "#FFFACD",
    "Elves": "#90EE90",
    "Dwarves": "#DEB887",
    "Rats": "#FFB6C1",
    "Spies": "#ADD8E6",
    "Companions": "#AFEEEE",
    "Villains": "#F5F5EE"
}

#przypisywanie postaci do grup
#musimy ręcznie mapować klucze z base_characters_map do kategorii
groups = {
    "Ciri": ["Ciri"],
    "Witchers": ["Vesemir", "Lambert", "Eskel", "Coen", "Geralt"],
    "Sorceresses": [
        "Yennefer", "Triss", "Philippa Eilhart", "Fringilla Vigo", "Keira Metz",
        "Margarita Laux-Antille", "Sheala de Tancarville", "Sabrina Glevissig",
        "Assire var Anahid", "Ida Emean aep Sivney", "Tissaia de Vries",
        "Lydia van Bredevoort", "Marti Sodergren", "Vilgefortz", "Stregobor",
        "Istredd", "Rience", "Schirru", "Avallac'h"
    ],
    "Rulers": [
        "Emhyr", "Foltest", "Meve", "Henselt", "Demavend", "Calanthe",
        "Pavetta", "Esterad Thyssen", "Anna Henrietta", "Auberon Muircetach",
        "Eredin", "Francesca Findabair"
    ],
    "Elves": ["Toruviel", "Yaevinn", "Isengrim Faoiltiarna", "Filavandrel", "Lara Dorren", "Ihuarraquax"],
    "Dwarves": ["Zoltan Chivay", "Yarpen Zigrin"],
    "Rats": ["Mistle", "Kayleigh", "Giselher", "Iskra", "Reef", "Asse"],
    "Spies": ["Sigismund Dijkstra", "Vattier de Rideaux", "Codringher", "Fenn"],
    "Companions": [
        "Dandelion", "Milva", "Regis", "Cahir", "Angouleme", "Nenneke",
        "Shani", "Iola", "Jarre", "Dudu", "Essi Daven", "Borch Three Jackdaws",
        "Crach an Craite", "Mousesack"
    ],
    "Villains": ["Bonhart", "Stefan Skellen", "Renfri"]
}

#tworzymy slownik
character_groups = {char: group for group, chars in groups.items() for char in chars}

def chart0():
    plt.figure(figsize=(20, 12))

    if G1.number_of_nodes() > 0:
        # Używamy tego samego seeda, żeby układ był w miarę powtarzalny
        pos = nx.spring_layout(G1, k=2.5, seed=42)

        # Pobieranie kolorów dla węzłów w G1
        node_colors_list = []
        for node in G1.nodes():
            group = character_groups.get(node)
            # Jeśli jakiejś postaci nie ma w grupach, dajemy kolor domyślny (szary)
            color = group_colors.get(group, "#CCCCCC")
            node_colors_list.append(color)

        # Rozmiar węzłów bazujący na centralności c1 (Basic)
        # Mnożnik 4000 taki sam jak w chart1 dla zachowania skali
        node_sizes = [c1[n] * 4000 for n in G1.nodes()]

        # Rysowanie węzłów
        nx.draw_networkx_nodes(G1, pos, node_size=node_sizes, node_color=node_colors_list, edgecolors='black')

        # Rysowanie krawędzi
        weights = [G1[u][v]['weight'] for u, v in G1.edges()]
        max_weight = max(weights) if weights else 1
        # Skalowanie grubości krawędzi
        widths = [(w / max_weight) * 4 + 0.3 for w in weights]

        nx.draw_networkx_edges(G1, pos, width=widths, edge_color='gray', alpha=0.4)

        # Etykiety
        nx.draw_networkx_labels(G1, pos, font_size=8, font_weight='bold')

        # Legenda (wspólna dla obu wykresów)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                                  markerfacecolor=color, markersize=10, markeredgecolor='black')
                           for key, color in group_colors.items()]

        plt.legend(handles=legend_elements, loc='upper left', title="Grupy Postaci", fontsize='small')

        plt.title("Sieć Interakcji - Model Podstawowy (Basic)", fontsize=14)
        plt.axis('off')

        plt.show()

#wykres pokazujacy polaczenia miedzy postaciami
def chart1():
    plt.figure(figsize=(20, 12))
    if G2.number_of_nodes() > 0:
        pos = nx.spring_layout(G2, k=2.5, seed=42)

        #dodajemy kolory dla nodow
        node_colors_list = []
        for node in G2.nodes():
            group = character_groups.get(node)
            color = group_colors.get(group)
            node_colors_list.append(color)

        #rozmiar wezlow
        node_sizes = [c2[n] * 4000 for n in G2.nodes()]

        #rysujemy wezly z kolorami
        nx.draw_networkx_nodes(G2, pos, node_size=node_sizes, node_color=node_colors_list, edgecolors='black')

        #rysujemy krawedzie
        weights = [G2[u][v]['weight'] for u, v in G2.edges()]
        max_weight = max(weights) if weights else 1
        widths = [(w / max_weight) * 4 + 0.3 for w in weights]
        nx.draw_networkx_edges(G2, pos, width=widths, edge_color='gray', alpha=0.4)

        #etykietujemy nody
        nx.draw_networkx_labels(G2, pos, font_size=8, font_weight='bold')

        #legenda
        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=color, markersize=10, markeredgecolor='black')
                       for key, color in group_colors.items()]

        plt.legend(handles=legend_elements, loc='upper left', title="Grupy Postaci", fontsize='small')

        plt.title("Ostateczna Sieć Interakcji (Kolorowanie wg Grup)", fontsize=14)
        plt.axis('off')

        plt.show()

#wykres zmiany centralnosci
def chart2():
    top_changes = sorted_diff[-10:]
    names = [x[0] for x in top_changes]
    values = [x[1] for x in top_changes]
    colors = ['red' if x < 0 else 'green' for x in values]

    plt.barh(names, values, color=colors)
    plt.title("Różnica Centralności (Context vs Basic)")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()

#graf roznic
def chart3():
    G_diff = nx.Graph()
    for (u, v), w in edge_diff.items():
        if abs(w) > 1:
            G_diff.add_edge(u, v, weight=w)

    if G_diff.number_of_nodes() > 0:
        pos_diff = nx.spring_layout(G_diff, k=2, seed=42)
        diff_edge_colors = ['green' if G_diff[u][v]['weight'] > 0 else 'red' for u, v in G_diff.edges()]
        diff_widths = [min(abs(G_diff[u][v]['weight']) * 0.3, 5) for u, v in G_diff.edges()]

        nx.draw_networkx_nodes(G_diff, pos_diff, node_size=200, node_color='lightgrey')
        nx.draw_networkx_labels(G_diff, pos_diff, font_size=7)
        nx.draw_networkx_edges(G_diff, pos_diff, edge_color=diff_edge_colors, width=diff_widths, alpha=0.6)
        plt.title("Zmiany w Krawędziach (>1)", fontsize=10)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
