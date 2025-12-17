"""
Generació d'un dataset sintètic de ciclistes del Port del Cantó
"""

import random
import pandas as pd


def generar_dataset(n=200, seed=42):
    random.seed(seed)

    tipus = {
        "BEBB": (3000, 1400),
        "BEMB": (3200, 2100),
        "MEBB": (4000, 1500),
        "MEMB": (4300, 2200)
    }

    data = []
    cid = 0

    for etiqueta, (tp_mitja, tb_mitja) in tipus.items():
        for _ in range(n // 4):
            tp = random.gauss(tp_mitja, 150)
            tb = random.gauss(tb_mitja, 120)
            tt = tp + tb
            data.append([cid, tp, tb, tt, etiqueta])
            cid += 1

    return pd.DataFrame(
        data,
        columns=["id", "tp", "tb", "tt", "tipus"]
    )


if __name__ == "__main__":
    df = generar_dataset()
    df.to_csv("data\\ciclistes.csv", index=False)
    print(df.head())
