"""
test_cache.py
-------------
Prueba el caché semántico.
"""

import time
from src.rag.chain import consultar

print("=" * 60)
print("PRUEBA DE CACHÉ SEMÁNTICO")
print("=" * 60)

def probar(pregunta: str):
    print(f"\nConsulta: '{pregunta}'")
    inicio = time.time()
    respuesta, desde_cache = consultar(pregunta)
    duracion = round(time.time() - inicio, 2)
    origen = "CACHÉ" if desde_cache else "GEMINI"
    print(f"Origen: {origen} | Tiempo: {duracion}s")
    print(f"Respuesta: {respuesta[:80]}...")
    print("-" * 60)

# Primera consulta — va a Gemini
probar("el tractor pierde aceite por el motor")

# Consulta idéntica — debería usar caché
probar("el tractor pierde aceite por el motor")

# Consulta similar — debería usar caché
probar("hay una fuga de aceite en el motor")

# Consulta diferente — va a Gemini
probar("el motor no arranca")