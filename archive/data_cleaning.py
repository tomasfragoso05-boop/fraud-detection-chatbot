"""
PaySim — Passo 1: Limpeza do dataset
======================================
Baseado na análise real do ficheiro:
  Synthetic_Financial_datasets_log.csv
  6.362.620 linhas · 11 colunas · 0 nulos · 0 duplicados
"""

import pandas as pd

# ── CARREGAR ──────────────────────────────────────────
df = pd.read_csv("Synthetic_Financial_datasets_log.csv")

print("=" * 55)
print("ANTES DA LIMPEZA")
print("=" * 55)
print(f"Linhas:  {len(df):,}")
print(f"Colunas: {df.shape[1]}")
print(f"Fraude:  {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")


# ── LIMPEZA 1: sem duplicados (confirmado) ────────────
# Verificado: 0 duplicados no ficheiro. Mantém-se por segurança.
antes = len(df)
df = df.drop_duplicates()
print(f"\n[1] Duplicados removidos: {antes - len(df)}")


# ── LIMPEZA 2: filtrar tipos com fraude ───────────────
# Fraude existe APENAS em TRANSFER (4097) e CASH_OUT (4116).
# PAYMENT, CASH_IN e DEBIT têm isFraud=0 em 100% dos casos.
#   CASH_OUT  -> 2.237.500 linhas
#   PAYMENT   -> 2.151.495 linhas  (zero fraude -- fora)
#   CASH_IN   -> 1.399.284 linhas  (zero fraude -- fora)
#   TRANSFER  ->   532.909 linhas
#   DEBIT     ->    41.432 linhas  (zero fraude -- fora)
antes = len(df)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
removidas = antes - len(df)
print(f"[2] Linhas removidas (PAYMENT, CASH_IN, DEBIT): {removidas:,}")
print(f"    Ficaram: {len(df):,} linhas")


# ── LIMPEZA 3: remover linhas com amount = 0 ──────────
# Encontradas 16 transacoes com amount=0 -- todas sao fraude (isFraud=1).
# Sao casos anomalos da simulacao: contas com saldo 0 a fazer CASH_OUT de 0.
# Nao representam padroes reais e podem confundir o modelo.
# Sao apenas 16 em 2.770.409 -- o impacto e nulo.
antes = len(df)
df = df[df["amount"] > 0].copy()
print(f"[3] Removidas linhas com amount=0: {antes - len(df)}")


# ── LIMPEZA 4: remover colunas inuteis ────────────────

# nameOrig / nameDest: IDs de contas (ex: "C1231006815", "M1979787155")
# Sao strings unicas -- o modelo nao aprende nada com elas diretamente.
# Para usar precisariam de feature engineering complexo (graph features, etc.)
df = df.drop(columns=["nameOrig", "nameDest"])
print("[4] Removidas: nameOrig, nameDest (IDs alfanumericos sem valor direto)")

# isFlaggedFraud: sistema interno do banco que tentou detetar fraude.
# Resultado: so 16 flags em 8213 fraudes reais (0.19% de detecao).
# E quasi-constante (quase sempre 0) -- nao discrimina nada.
df = df.drop(columns=["isFlaggedFraud"])
print("[5] Removida: isFlaggedFraud (16 flags em 8213 fraudes -- inutil)")

# step: hora da simulacao (1 a 743 = 30 dias).
# A distribuicao por hora e quase identica em fraude e nao-fraude.
# Nao ajuda o modelo a separar as duas classes.
df = df.drop(columns=["step"])
print("[6] Removida: step (distribuicao igual em fraude e legitimo)")


# ── RESULTADO FINAL ───────────────────────────────────
print("\n" + "=" * 55)
print("DEPOIS DA LIMPEZA")
print("=" * 55)
print(f"Linhas:   {len(df):,}")
print(f"Colunas:  {df.shape[1]}")
print(f"\nColunas que ficaram:")
print(df.dtypes.to_string())
print(f"\nFraude:   {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")
print(f"\nDistribuicao por tipo:")
print(df["type"].value_counts().to_string())
print(f"\nPrimeiras 3 linhas:")
print(df.head(3).to_string())


# ── GUARDAR ───────────────────────────────────────────
df.to_csv("transactions_limpo.csv", index=False)
print("\nFicheiro guardado: transactions_limpo.csv")
print("Proximo passo: feature engineering")
