# Chunking del manual SINUMERIK 808D ADVANCED

Documentación del proceso de división en chunks del manual de diagnósticos para su uso en un sistema RAG.

---

## 1. Datos de entrada

| Atributo | Valor |
|---|---|
| Fichero | `808D_ADV_diagnostics_man_0718_en-US.md` |
| Origen | Manual PDF convertido a Markdown |
| Tamaño | 1,352,639 caracteres — 35,401 líneas |
| Contenido | ~500 páginas: prefacio, 9 capítulos y apéndice |

El documento contiene principalmente **entradas de alarma NC/PLC**, cada una con estructura fija:

```
# <número> <nombre de la alarma>

**Parameters:**            ...
**Explanation:**           ...
**Reaction:**              ...
**Remedy:**                ...
**Programm continuation:** ...
```

Sin embargo, el formato no es completamente homogéneo: algunas alarmas usan encabezado `#`, otras `##`, y en la sección PLC el número aparece en negrita (`**400000** **descripción**`). El documento también incluye tablas HTML, referencias a imágenes y separadores horizontales (`---`, `***`) heredados del parser de PDF.

---

## 2. Evaluación inicial de estrategias

Se compararon cuatro estrategias de chunking mediante el script `chunker_analysis.py`:

| # | Estrategia | Descripción |
|---|---|---|
| 1 | **Alarm-based** | Un chunk por entrada de alarma (split por regex sobre el patrón de número de alarma) |
| 2 | **Header-based (h1+h2)** | Split en encabezados Markdown `#` y `##` |
| 3 | **Separator-based** | Split en separadores horizontales `---` y `***` |
| 4 | **Recursive (1500 ch)** | Split recursivo en cascada por separadores, con tamaño objetivo de 1500 chars y solapamiento de 150 |

![Comparativa de estrategias de chunking](img/chunk_analysis.png)

### Resultados comparativos

| Métrica | 1 Alarm-based | 2 Header-based | 3 Separator-based | 4 Recursive |
|---|---:|---:|---:|---:|
| N.º chunks | 1,649 | 2,002 | 426 | 1,122 |
| Media | 818 | 674 | 3,168 | 1,345 |
| Mediana | 565 | 490 | 708 | 1,402 |
| Mínimo | 130 | 6 | 173 | 354 |
| Máximo | 85,117 | 5,042 | 62,155 | 1,651 |
| P5 | 267 | 17 | 264 | 856 |
| P95 | 1,461 | 2,342 | 12,474 | 1,639 |

### Conclusiones de la evaluación

- **Estrategia 2** genera muchos chunks inútiles (mínimo de 6 chars), fruto de encabezados huérfanos del parser de PDF.
- **Estrategia 3** tiene varianza extrema (mediana 708 vs. máximo 62,155) porque los separadores `---`/`***` solo aparecen en algunas secciones.
- **Estrategia 4** produce la distribución más uniforme pero es ciega a la semántica: corta alarmas en mitad de sus campos.
- **Estrategia 1** es la más apropiada para este documento: la alarma individual es la unidad semántica natural para un sistema RAG. Su único problema es un 2.4% de chunks que superan los 2,000 caracteres (alarmas con explicaciones muy extensas), que se resuelve en el paso 2.

---

## 3. Estrategia final

La estrategia final, implementada en `chunker.py`, combina la estrategia 1 con una subdivisión recursiva para los chunks grandes.

### Paso 1 — Alarm-based split

Se divide el documento completo usando una expresión regular con **lookahead** sobre los tres formatos de encabezado de alarma:

```python
pattern = re.compile(
    r'(?=^#{1,3}\s+\d{4,6}[\s%]|^\*\*\d{4,6}\*\*)',
    re.MULTILINE
)
```

El lookahead garantiza que el delimitador queda dentro del chunk (no se pierde el encabezado de la alarma). Resultado: **1,649 chunks**, de los cuales **40 (2.4%) superan los 2,000 caracteres**.

### Paso 2 — Subdivisión recursiva de chunks grandes (> 2,000 chars)

Para los 40 chunks que superan el umbral se aplica una subdivisión distinta según si el chunk tiene o no estructura de alarma.

**Chunks con estructura de alarma:**

1. Se intenta dividir primero en los **límites de campo** semánticos de la alarma (`**Explanation:**`, `**Reaction:**`, `**Remedy:**`, `**Programm continuation:**`), que son las fronteras más naturales dentro de una entrada.
2. Si algún campo sigue siendo demasiado grande, se aplica un **splitter recursivo en cascada** con prioridad decreciente de separadores: párrafo (`\n\n`) → línea (`\n`) → palabra (` `).
3. **Preservación del contexto:** cada sub-chunk de continuación recibe el encabezado de la alarma original inyectado como prefijo, más los últimos 150 caracteres del sub-chunk anterior como solapamiento. Esto garantiza que un sub-chunk recuperado de forma aislada siempre identifica a qué alarma pertenece:

```
## 25000 Axis %1 hardware fault of active encoder [cont. 1/2]

[...] ...replace the encoder if faults are found.
Monitoring can be switched off by setting MD36310...

**Programm continuation:**
Switch control OFF - ON.
```

**Chunks sin estructura de alarma** (prefacio, introducciones de capítulo): se aplica directamente el splitter recursivo en cascada sin inyección de cabecera.

### Resultados finales

![Distribución final de longitudes de chunk](img/chunk_final_analysis.png)

| Métrica | Valor |
|---|---:|
| N.º chunks totales | 1,812 |
| Media | 758 chars |
| Mediana | 592 chars |
| Mínimo | 102 chars |
| Máximo | 2,269 chars |
| P5 | 265 chars |
| P95 | 1,925 chars |
| Chunks > 2,000 chars | 59 (3.3%) |

Los 59 chunks que superan el umbral tras el paso 2 son líneas individuales sin separadores naturales (URLs largas, cadenas técnicas) cuyo exceso es mínimo (máximo 2,269 chars). Los sub-chunks de continuación pueden superar ligeramente 2,000 chars cuando el solapamiento y el prefijo de contexto inyectado empujan el tamaño por encima del umbral.

---

## 4. Datos de salida

| Atributo | Valor |
|---|---|
| Fichero | `808D_ADV_diagnostics_man_0718_en-US_chunked.md` |
| Tamaño | 1.4 MB |
| N.º chunks | 1,812 |
| Separador entre chunks | `\n<CHUNK>\n` |
| Codificación | UTF-8 |

El fichero puede leerse y reconstituirse en una lista de chunks con:

```python
chunks = Path("808D_ADV_diagnostics_man_0718_en-US_chunked.md") \
             .read_text(encoding="utf-8") \
             .split("\n<CHUNK>\n")
```
