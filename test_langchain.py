"""
test_langchain.py
-----------------
Prueba el pipeline RAG con memoria conversacional.
Mide tokens y tiempos por turno.
"""

import time
from src.rag.chain import consultar, llm, historial

print("=" * 60)
print("CONVERSACIÓN CON MEMORIA")
print("=" * 60)

preguntas = [
    "el tractor pierde aceite por el motor",
    "cuántas unidades necesito del primero?",
    "y cuánto cuesta aproximadamente?",
]

for i, pregunta in enumerate(preguntas, start=1):
    print(f"\nTurno {i}")
    print(f"Usuario: {pregunta}")

    inicio = time.time()
    respuesta = consultar(pregunta)
    duracion = round(time.time() - inicio, 2)

    print(f"Sistema: {respuesta}")
    print(f"Tiempo: {duracion}s | Mensajes en historial: {len(historial)}")
    print("-" * 60)