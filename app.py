import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import altair as alt
import random
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

st.set_page_config(page_title="CalcUMG", page_icon="🧮", layout="wide")

x = sp.symbols("x")
TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def leer_funcion(texto):
    """Convierte el texto escrito por el usuario en una función de sympy."""
    if texto.strip() == "":
        return None, "Escribe una función para continuar."
    try:
        f = parse_expr(texto, local_dict={"x": x}, transformations=TRANSFORMACIONES)
    except Exception:
        return None, "No pude entender la función. Ejemplo válido: x^3 - 6*x^2 + 9*x + 1"
    if f.free_symbols - {x}:
        return None, "Solo puedo trabajar con la variable x."
    return f, None


def modulo_derivadas():
    st.header("Derivadas")
    st.write("Escribe una función de x y te muestro la derivada paso a paso.")
    texto = st.text_input("f(x) =", value="x^3 - 6*x^2 + 9*x + 1")

    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return
    if not st.button("Calcular derivada"):
        return

    st.subheader("1. Función")
    st.latex("f(x) = " + sp.latex(f))

    st.subheader("2. Método")
    st.write("Derivamos término por término (regla de la suma) usando las reglas básicas:")
    st.latex(r"\frac{d}{dx}\left(x^n\right) = n\,x^{n-1}")

    st.subheader("3. Derivada de cada término")
    for t in sp.Add.make_args(sp.expand(f)):
        st.latex(
            r"\frac{d}{dx}\left(" + sp.latex(t) + r"\right) = " + sp.latex(sp.diff(t, x))
        )

    d1 = sp.simplify(sp.diff(f, x))
    st.subheader("4. Respuesta final")
    st.latex("f'(x) = " + sp.latex(d1))

    st.subheader("5. Verificación numérica")
    x0, h = 2, 1e-5
    try:
        aprox = (float(f.subs(x, x0 + h)) - float(f.subs(x, x0 - h))) / (2 * h)
        exacto = float(d1.subs(x, x0))
        st.write(f"En x = {x0}: cociente incremental ≈ {aprox:.4f} y f'({x0}) = {exacto:.4f}")
    except Exception:
        st.info("No se pudo verificar en x = 2 (la función podría no estar definida ahí).")

def app_extremos():
    st.header("Aplicaciones: puntos críticos y extremos")
    st.write("Encuentra los puntos críticos de f(x) y clasifica los extremos relativos con el criterio de la segunda derivada.")
    texto = st.text_input("f(x) =", value="x^3 - 6*x^2 + 9*x + 1", key="app_f")

    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return
    if not st.button("Analizar función", key="app_btn"):
        return

    st.subheader("1. Función")
    st.latex("f(x) = " + sp.latex(f))

    d1 = sp.simplify(sp.diff(f, x))
    st.subheader("2. Primera derivada")
    st.latex("f'(x) = " + sp.latex(d1))

    st.subheader("3. Puntos críticos")
    st.write("Los puntos críticos se encuentran resolviendo f'(x) = 0.")
    st.latex(sp.latex(d1) + " = 0")
    try:
        criticos = [c for c in sp.solve(sp.Eq(d1, 0), x) if c.is_real]
    except Exception:
        st.error("No pude resolver f'(x) = 0 para esta función.")
        return
    if not criticos:
        st.info("Esta función no tiene puntos críticos reales.")
        return
    for c in criticos:
        st.latex("x = " + sp.latex(c))

    d2 = sp.simplify(sp.diff(d1, x))
    st.subheader("4. Segunda derivada")
    st.latex("f''(x) = " + sp.latex(d2))

    st.subheader("5. Clasificación de extremos")
    st.write("Si f''(c) > 0 hay un mínimo relativo. Si f''(c) < 0 hay un máximo relativo.")
    for c in criticos:
        y = sp.simplify(f.subs(x, c))
        v = sp.simplify(d2.subs(x, c))
        st.latex(
            "f(" + sp.latex(c) + ") = " + sp.latex(y)
            + r"\qquad f''(" + sp.latex(c) + ") = " + sp.latex(v)
        )
        punto = "(" + sp.latex(c) + ", " + sp.latex(y) + ")"
        if v.is_positive:
            st.success("Mínimo relativo en el punto")
            st.latex(punto)
        elif v.is_negative:
            st.success("Máximo relativo en el punto")
            st.latex(punto)
        else:
            st.warning("El criterio de la segunda derivada no decide en este punto (f'' = 0). Hay que usar el criterio de la primera derivada.")
def modulo_graficas():
    st.header("Representación gráfica")
    st.write("Grafica f(x) y marca sus intersecciones y puntos importantes.")
    texto = st.text_input("f(x) =", value="x^3 - 6*x^2 + 9*x + 1", key="graf_f")
    col1, col2 = st.columns(2)
    xmin = col1.number_input("x mínimo", value=-2.0, key="graf_xmin")
    xmax = col2.number_input("x máximo", value=6.0, key="graf_xmax")

    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return
    if xmin >= xmax:
        st.warning("El valor mínimo debe ser menor que el máximo.")
        return
    if not st.button("Graficar", key="graf_btn"):
        return

    try:
        fn = sp.lambdify(x, f, "numpy")
        xs = np.linspace(xmin, xmax, 600)
        with np.errstate(all="ignore"):
            ys = np.array(fn(xs), dtype=float)
            if ys.ndim == 0:
                ys = np.full_like(xs, float(ys))
            ys[np.abs(ys) > 1e6] = np.nan
    except Exception:
        st.error("No pude graficar esta función en ese intervalo.")
        return

    def reales_en_rango(ecuacion):
        try:
            soluciones = sp.solve(ecuacion, x)
        except Exception:
            return []
        resultado = []
        for s in soluciones:
            if s.is_real and xmin <= float(s) <= xmax:
                resultado.append(s)
        return resultado

    d1 = sp.diff(f, x)
    cortes_x = reales_en_rango(f)
    criticos = reales_en_rango(d1)

    filas = []
    for c in cortes_x:
        filas.append({"x": float(c), "y": 0.0, "tipo": "Corte con eje x"})
    for c in criticos:
        try:
            filas.append({"x": float(c), "y": float(f.subs(x, c)), "tipo": "Punto crítico"})
        except Exception:
            pass
    if xmin <= 0 <= xmax:
        try:
            filas.append({"x": 0.0, "y": float(f.subs(x, 0)), "tipo": "Corte con eje y"})
        except Exception:
            pass

    linea = alt.Chart(pd.DataFrame({"x": xs, "y": ys})).mark_line().encode(x="x:Q", y="y:Q")
    grafica = linea
    if filas:
        puntos = alt.Chart(pd.DataFrame(filas)).mark_point(size=120, filled=True).encode(
            x="x:Q", y="y:Q", color="tipo:N", tooltip=["tipo", "x", "y"]
        )
        grafica = linea + puntos
    st.altair_chart(grafica.interactive(), width="stretch")

    st.subheader("Interpretación")
    if cortes_x:
        st.write("Intersecciones con el eje x (raíces):")
        for c in cortes_x:
            st.latex("x = " + sp.latex(c))
    else:
        st.write("No hay intersecciones con el eje x en este intervalo.")

    if xmin <= 0 <= xmax:
        try:
            st.write(f"Intersección con el eje y: f(0) = {float(f.subs(x, 0)):.4f}")
        except Exception:
            st.write("La función no está definida en x = 0.")

    st.write("Crecimiento de la función:")
    puntos_x = sorted([float(c) for c in criticos])
    limites = [xmin] + puntos_x + [xmax]
    for a, b in zip(limites[:-1], limites[1:]):
        if b - a < 1e-9:
            continue
        try:
            pendiente = float(d1.subs(x, (a + b) / 2))
        except Exception:
            continue
        if pendiente > 0:
            estado = "creciente"
        elif pendiente < 0:
            estado = "decreciente"
        else:
            estado = "constante"
        st.write(f"- En el intervalo ({a:.2f}, {b:.2f}) la función es **{estado}**.")
def alg_resolver():
    st.write("Escribe una ecuación con un solo signo =, por ejemplo: x^2 - 5*x + 6 = 0")
    texto = st.text_input("Ecuación:", value="x^2 - 5*x + 6 = 0", key="alg_eq")
    if not st.button("Resolver ecuación", key="alg_btn_res"):
        return
    if texto.count("=") != 1:
        st.warning("La ecuación debe tener exactamente un signo =. Ejemplo: x^2 - 4 = 0")
        return
    texto_izq, texto_der = texto.split("=")
    izq, error1 = leer_funcion(texto_izq)
    der, error2 = leer_funcion(texto_der)
    if error1 or error2:
        st.warning(error1 or error2)
        return

    st.subheader("1. Ecuación")
    st.latex(sp.latex(izq) + " = " + sp.latex(der))

    expr = sp.expand(izq - der)
    st.subheader("2. Pasamos todo a un lado")
    st.latex(sp.latex(expr) + " = 0")

    if x not in expr.free_symbols:
        if expr == 0:
            st.info("La igualdad se cumple siempre: hay infinitas soluciones.")
        else:
            st.info("La ecuación no tiene solución.")
        return

    st.subheader("3. Método")
    if expr.is_polynomial(x) and sp.degree(expr, x) == 1:
        a, b = sp.Poly(expr, x).all_coeffs()
        st.write("Es una ecuación lineal a·x + b = 0, entonces x = -b / a.")
        st.latex("a = " + sp.latex(a) + r",\quad b = " + sp.latex(b))
        st.latex(r"x = \frac{-(" + sp.latex(b) + r")}{" + sp.latex(a) + "}")
    elif expr.is_polynomial(x) and sp.degree(expr, x) == 2:
        a, b, c = sp.Poly(expr, x).all_coeffs()
        st.write("Es una ecuación cuadrática a·x² + b·x + c = 0. Usamos la fórmula general:")
        st.latex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
        st.latex("a = " + sp.latex(a) + r",\quad b = " + sp.latex(b) + r",\quad c = " + sp.latex(c))
        st.write("Sustituyendo los valores:")
        st.latex(
            r"x = \frac{-(" + sp.latex(b) + r") \pm \sqrt{(" + sp.latex(b) + r")^2 - 4("
            + sp.latex(a) + r")(" + sp.latex(c) + r")}}{2(" + sp.latex(a) + ")}"
        )
        disc = b**2 - 4 * a * c
        st.latex(r"\Delta = b^2 - 4ac = " + sp.latex(disc))
        if disc > 0:
            st.write("Δ > 0: hay dos soluciones reales distintas.")
        elif disc == 0:
            st.write("Δ = 0: hay una solución real (doble).")
        else:
            st.write("Δ < 0: no hay soluciones reales (las soluciones son complejas).")
    else:
        st.write("Resolvemos la ecuación algebraicamente. Si es un polinomio, mostramos también su factorización:")
        if expr.is_polynomial(x):
            st.latex(sp.latex(expr) + " = " + sp.latex(sp.factor(expr)) + " = 0")

    st.subheader("4. Soluciones")
    try:
        soluciones = sp.solve(sp.Eq(izq, der), x)
    except Exception:
        st.error("No pude resolver esta ecuación.")
        return
    reales = [s for s in soluciones if s.is_real]
    if not reales:
        st.info("No encontré soluciones reales para esta ecuación.")
        return
    for s in reales:
        st.latex("x = " + sp.latex(s))
    if expr.has(sp.sin, sp.cos, sp.tan):
        st.info("Las funciones trigonométricas tienen infinitas soluciones; aquí se muestran las principales.")

    st.subheader("5. Verificación")
    for s in reales:
        try:
            izq_val = sp.simplify(izq.subs(x, s))
            der_val = sp.simplify(der.subs(x, s))
            st.latex("x = " + sp.latex(s) + r":\quad " + sp.latex(izq_val) + " = " + sp.latex(der_val))
        except Exception:
            pass


def alg_factorizar():
    st.write("Escribe una expresión y la factorizo.")
    texto = st.text_input("Expresión:", value="x^3 - 6*x^2 + 11*x - 6", key="alg_fact")
    if not st.button("Factorizar", key="alg_btn_fact"):
        return
    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return

    st.subheader("1. Expresión")
    st.latex(sp.latex(f))

    fact = sp.factor(f)
    st.subheader("2. Factorización")
    st.write("Se aplican técnicas como factor común, diferencia de cuadrados, trinomios y agrupación.")
    st.latex(sp.latex(f) + " = " + sp.latex(fact))
    if fact == f:
        st.info("Esta expresión ya está factorizada o no se puede factorizar más con números enteros.")

    st.subheader("3. Verificación")
    st.write("Si multiplicamos los factores, obtenemos otra vez la expresión original:")
    st.latex(sp.latex(fact) + " = " + sp.latex(sp.expand(fact)))


def alg_dominio_rango():
    st.write("Escribe una función y calculo su dominio y su rango.")
    texto = st.text_input("f(x) =", value="sqrt(x - 2)", key="alg_dom")
    if not st.button("Calcular dominio y rango", key="alg_btn_dom"):
        return
    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return

    st.subheader("1. Función")
    st.latex("f(x) = " + sp.latex(f))

    st.subheader("2. Restricciones")
    restricciones = []
    for p in f.atoms(sp.Pow):
        base, exponente = p.as_base_exp()
        if base.has(x):
            if exponente.is_negative:
                restricciones.append(sp.latex(base) + r" \neq 0")
            if exponente.is_Rational and exponente.q % 2 == 0:
                restricciones.append(sp.latex(base) + r" \geq 0")
    for lg in f.atoms(sp.log):
        restricciones.append(sp.latex(lg.args[0]) + " > 0")
    restricciones = list(dict.fromkeys(restricciones))
    if restricciones:
        st.write("Condiciones para x (denominador distinto de cero, raíz con argumento ≥ 0, logaritmo con argumento > 0):")
        for r in restricciones:
            st.latex(r)
    else:
        st.write("No hay denominadores, raíces pares ni logaritmos que restrinjan a x.")

    st.subheader("3. Dominio")
    dominio = None
    try:
        dominio = continuous_domain(f, x, sp.S.Reals)
        st.latex(r"\text{Dom}(f) = " + sp.latex(dominio))
    except Exception:
        st.warning("No pude calcular el dominio de esta función.")

    st.subheader("4. Rango")
    try:
        rango = function_range(f, x, dominio if dominio is not None else sp.S.Reals)
        st.latex(r"\text{Rango}(f) = " + sp.latex(rango))
    except Exception:
        st.warning("No pude calcular el rango de esta función. Prueba con una más sencilla.")


def alg_composicion():
    st.write("Escribe dos funciones y calculo f(g(x)) y g(f(x)).")
    col1, col2 = st.columns(2)
    texto_f = col1.text_input("f(x) =", value="x^2 + 1", key="alg_f")
    texto_g = col2.text_input("g(x) =", value="2*x - 3", key="alg_g")
    if not st.button("Componer", key="alg_btn_comp"):
        return
    f, error1 = leer_funcion(texto_f)
    g, error2 = leer_funcion(texto_g)
    if error1 or error2:
        st.warning(error1 or error2)
        return

    st.subheader("1. Funciones")
    st.latex("f(x) = " + sp.latex(f))
    st.latex("g(x) = " + sp.latex(g))

    st.subheader("2. Composición f(g(x))")
    st.write("Sustituimos g(x) en lugar de x dentro de f:")
    fog = f.xreplace({x: g})
    st.latex("f(g(x)) = " + sp.latex(fog))
    st.write("Desarrollando:")
    st.latex("f(g(x)) = " + sp.latex(sp.expand(fog)))

    st.subheader("3. Composición g(f(x))")
    st.write("Sustituimos f(x) en lugar de x dentro de g:")
    gof = g.xreplace({x: f})
    st.latex("g(f(x)) = " + sp.latex(gof))
    st.write("Desarrollando:")
    st.latex("g(f(x)) = " + sp.latex(sp.expand(gof)))


def modulo_algebra():
    st.header("Álgebra y funciones")
    opcion = st.radio(
        "¿Qué quieres hacer?",
        ["Resolver ecuación", "Factorizar", "Dominio y rango", "Composición de funciones"],
        horizontal=True,
    )
    if opcion == "Resolver ecuación":
        alg_resolver()
    elif opcion == "Factorizar":
        alg_factorizar()
    elif opcion == "Dominio y rango":
        alg_dominio_rango()
    else:
        alg_composicion()         
def modulo_limites():
    st.header("Límites")
    st.write("Calcula el límite de f(x) cuando x tiende a un valor y explica el método utilizado.")
    texto = st.text_input("f(x) =", value="(x^2 - 4)/(x - 2)", key="lim_f")
    col1, col2 = st.columns(2)
    texto_a = col1.text_input(
        "x tiende a (usa oo para infinito, -oo para menos infinito):", value="2", key="lim_a"
    )
    lado = col2.radio(
        "Dirección:", ["Ambos lados", "Por la izquierda", "Por la derecha"], key="lim_lado"
    )

    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return
    try:
        a = parse_expr(texto_a.strip(), transformations=standard_transformations + (convert_xor,))
    except Exception:
        st.warning("No entendí el valor al que tiende x. Ejemplos: 2, -1, 1/2, oo, -oo")
        return
    infinito = a in (sp.oo, -sp.oo)
    if not infinito and not (a.is_number and a.is_real):
        st.warning("El valor al que tiende x debe ser un número real, oo o -oo.")
        return
    if not st.button("Calcular límite", key="lim_btn"):
        return

    # ----- 1. Planteamiento -----
    st.subheader("1. Límite a calcular")
    flecha = r"x \to " + sp.latex(a)
    if not infinito and lado == "Por la izquierda":
        flecha += "^-"
    elif not infinito and lado == "Por la derecha":
        flecha += "^+"
    st.latex(r"\lim_{" + flecha + "} " + sp.latex(f))

    # ----- 2. Método -----
    st.subheader("2. Método")
    try:
        if not infinito:
            valor = f.subs(x, a)
            st.write("Primero intentamos **sustitución directa**: reemplazamos x por el valor al que tiende.")
            st.latex("f(" + sp.latex(a) + ") = " + sp.latex(valor))
            if valor.is_real and valor.is_finite:
                st.write("El resultado es un número real, así que el límite es simplemente f(a).")
            elif valor in (sp.zoo, sp.oo, -sp.oo):
                st.write(
                    "El denominador se anula y el numerador no: el valor crece sin límite. "
                    "Hay que analizar los límites laterales (izquierda y derecha)."
                )
            elif valor.has(sp.nan):
                st.write(
                    "Obtenemos una **forma indeterminada** (como 0/0). La sustitución directa no sirve, "
                    "así que factorizamos y simplificamos:"
                )
                num, den = sp.fraction(sp.together(f))
                st.latex(
                    r"f(x) = \frac{" + sp.latex(sp.factor(num)) + "}{" + sp.latex(sp.factor(den)) + "}"
                )
                simpl = sp.cancel(f)
                st.write("Cancelamos los factores comunes (válido para x ≠ a):")
                st.latex("f(x) = " + sp.latex(simpl))
                nuevo = simpl.subs(x, a)
                if nuevo.is_real and nuevo.is_finite:
                    st.write("Ahora sí podemos sustituir:")
                    st.latex("f(" + sp.latex(a) + ") = " + sp.latex(nuevo))
                else:
                    st.write(
                        "Seguimos con forma indeterminada. Aplicamos la **regla de L'Hôpital**: "
                        "derivamos numerador y denominador por separado."
                    )
                    st.latex(
                        r"\lim \frac{" + sp.latex(sp.diff(num, x)) + "}{" + sp.latex(sp.diff(den, x)) + "}"
                    )
            else:
                st.write("La función no toma un valor real en ese punto, así que analizamos los límites laterales.")
        else:
            st.write("Como x tiende a infinito, comparamos el comportamiento de los términos de mayor potencia.")
            num, den = sp.fraction(sp.together(f))
            if num.is_polynomial(x) and den.is_polynomial(x):
                gn, gd = sp.degree(num, x), sp.degree(den, x)
                st.write(f"Es una función racional. Grado del numerador = {gn}, grado del denominador = {gd}.")
                if gn < gd:
                    st.write("Grado del numerador < grado del denominador: el límite es 0.")
                elif gn == gd:
                    st.write("Grados iguales: el límite es el cociente de los coeficientes principales.")
                else:
                    st.write("Grado del numerador > grado del denominador: el límite es infinito (±∞ según el signo).")
            else:
                st.write("Analizamos qué término domina cuando x crece (por ejemplo, una exponencial domina sobre un polinomio).")
    except Exception:
        st.info("No pude detallar el método para esta función, pero calculo el límite igualmente.")

    # ----- 3. Cálculo -----
    st.subheader("3. Cálculo del límite")
    try:
        if infinito:
            final = sp.limit(f, x, a)
            st.latex(r"\lim_{x \to " + sp.latex(a) + "} f(x) = " + sp.latex(final))
        else:
            izq = sp.limit(f, x, a, dir="-")
            der = sp.limit(f, x, a, dir="+")
            st.latex(r"\lim_{x \to " + sp.latex(a) + r"^-} f(x) = " + sp.latex(izq))
            st.latex(r"\lim_{x \to " + sp.latex(a) + r"^+} f(x) = " + sp.latex(der))
            if lado == "Por la izquierda":
                final = izq
            elif lado == "Por la derecha":
                final = der
            elif izq == der:
                final = izq
            else:
                final = None
    except Exception:
        st.error("No pude calcular este límite.")
        return

    # ----- 4. Respuesta final -----
    st.subheader("4. Respuesta final")
    if final is None:
        st.error("El límite NO existe, porque los límites laterales son distintos.")
    elif isinstance(final, (sp.Limit, sp.AccumBounds)):
        st.warning("El límite no existe o no pude determinarlo (la función oscila o es demasiado compleja).")
    else:
        st.success("Resultado:")
        st.latex(r"\lim_{" + flecha + "} f(x) = " + sp.latex(final))
        if final in (sp.oo, -sp.oo):
            st.write("El límite es infinito: la función crece (o decrece) sin límite, no se acerca a un número real.")

    # ----- 5. Verificación numérica -----
    st.subheader("5. Verificación numérica")

    def evaluar(v):
        try:
            return f"{float(f.subs(x, v)):.6f}"
        except Exception:
            return "no definido"

    if infinito:
        signo = 1 if a == sp.oo else -1
        valores = [signo * 10**k for k in (1, 2, 3, 4, 5)]
        st.table(pd.DataFrame({"x": [str(v) for v in valores], "f(x)": [evaluar(v) for v in valores]}))
    else:
        af = float(a)
        distancias = [0.1, 0.01, 0.001, 0.0001]
        c1, c2 = st.columns(2)
        c1.write("Acercándonos por la izquierda:")
        c1.table(pd.DataFrame({
            "x": [f"{af - d:.4f}" for d in distancias],
            "f(x)": [evaluar(af - d) for d in distancias],
        }))
        c2.write("Acercándonos por la derecha:")
        c2.table(pd.DataFrame({
            "x": [f"{af + d:.4f}" for d in distancias],
            "f(x)": [evaluar(af + d) for d in distancias],
        }))

def modulo_integrales():
    st.header("Integrales")
    st.write("Resuelve integrales indefinidas y definidas, mostrando el procedimiento.")
    tipo = st.radio("Tipo de integral:", ["Indefinida", "Definida"], horizontal=True, key="int_tipo")
    texto = st.text_input("f(x) =", value="3*x^2 + 2*x + 1", key="int_f")
    if tipo == "Definida":
        col1, col2 = st.columns(2)
        texto_a = col1.text_input("Límite inferior a:", value="0", key="int_a")
        texto_b = col2.text_input("Límite superior b:", value="2", key="int_b")

    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return

    if tipo == "Definida":
        try:
            trans = standard_transformations + (convert_xor,)
            a = parse_expr(texto_a.strip(), transformations=trans)
            b = parse_expr(texto_b.strip(), transformations=trans)
        except Exception:
            st.warning("No entendí los límites de integración. Ejemplos: 0, 2, -1, 1/2, pi")
            return
        if not (a.is_number and a.is_real and b.is_number and b.is_real):
            st.warning("Los límites de integración deben ser números reales (finitos).")
            return

    if not st.button("Calcular integral", key="int_btn"):
        return

    # ----- 1. Planteamiento -----
    st.subheader("1. Integral a resolver")
    if tipo == "Indefinida":
        st.latex(r"\int " + sp.latex(f) + r"\, dx")
    else:
        st.latex(r"\int_{" + sp.latex(a) + "}^{" + sp.latex(b) + "} " + sp.latex(f) + r"\, dx")

    # ----- 2. Método -----
    st.subheader("2. Método")
    st.write("Integramos término por término (regla de la suma) con las fórmulas básicas:")
    st.latex(r"\int x^n\,dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)")
    st.latex(r"\int \frac{1}{x}\,dx = \ln|x| + C \qquad \int e^x\,dx = e^x + C")
    st.latex(r"\int \sin x\,dx = -\cos x + C \qquad \int \cos x\,dx = \sin x + C")
    st.caption("Nota: el programa escribe el logaritmo natural como log(x).")

    # ----- 3. Cada término -----
    try:
        F = sp.integrate(f, x)
    except Exception:
        st.error("No pude resolver esta integral.")
        return
    if F.has(sp.Integral):
        st.error("No encontré una antiderivada elemental para esta función. Prueba con otra más sencilla.")
        return

    st.subheader("3. Integral de cada término")
    for t in sp.Add.make_args(sp.expand(f)):
        st.latex(r"\int " + sp.latex(t) + r"\,dx = " + sp.latex(sp.integrate(t, x)))

    # ----- Indefinida -----
    if tipo == "Indefinida":
        st.subheader("4. Respuesta final")
        st.latex(r"\int " + sp.latex(f) + r"\,dx = " + sp.latex(F) + " + C")

        st.subheader("5. Verificación")
        st.write("Derivamos el resultado: debemos obtener otra vez la función original.")
        dF = sp.simplify(sp.diff(F, x))
        st.latex(r"\frac{d}{dx}\left(" + sp.latex(F) + r" + C\right) = " + sp.latex(dF))
        try:
            if sp.simplify(dF - f) == 0:
                st.success("Coincide con f(x): la integral es correcta.")
            else:
                st.info("No pude comprobar la igualdad automáticamente; compara ambas expresiones.")
        except Exception:
            pass
        return

    # ----- Definida -----
    try:
        for s in sp.singularities(f, x):
            if s.is_real and min(float(a), float(b)) <= float(s) <= max(float(a), float(b)):
                st.error(
                    "La función no está definida en x = " + str(s)
                    + ", que está dentro del intervalo. Es una integral impropia y este asistente no la resuelve."
                )
                return
    except Exception:
        pass

    try:
        Fb = F.subs(x, b)
        Fa = F.subs(x, a)
        resultado = sp.simplify(sp.integrate(f, (x, a, b)))
    except Exception:
        st.error("No pude evaluar la integral definida.")
        return
    if resultado.has(sp.oo, sp.zoo, sp.nan) or resultado.has(sp.Integral):
        st.error("No pude evaluar esta integral en el intervalo dado (puede ser impropia).")
        return

    st.subheader("4. Teorema fundamental del cálculo")
    st.write("Si F es una antiderivada de f, entonces la integral definida es F(b) − F(a).")
    st.latex("F(x) = " + sp.latex(F))
    st.latex(r"\int_a^b f(x)\,dx = F(b) - F(a)")
    st.write("Sustituyendo los límites:")
    st.latex(r"= \left(" + sp.latex(Fb) + r"\right) - \left(" + sp.latex(Fa) + r"\right)")

    st.subheader("5. Respuesta final")
    st.latex(
        r"\int_{" + sp.latex(a) + "}^{" + sp.latex(b) + "} " + sp.latex(f)
        + r"\,dx = " + sp.latex(resultado)
    )

    st.subheader("6. Verificación numérica y gráfica")
    try:
        valor = float(resultado)
        fn = sp.lambdify(x, f, "numpy")
        xs = np.linspace(float(a), float(b), 2001)
        with np.errstate(all="ignore"):
            ys = np.array(fn(xs), dtype=float)
            if ys.ndim == 0:
                ys = np.full_like(xs, float(ys))
        aprox = float(np.sum((ys[:-1] + ys[1:]) / 2 * np.diff(xs)))
        st.write(f"Resultado exacto ≈ {valor:.6f}. Aproximación con trapecios ≈ {aprox:.6f} (deben ser casi iguales).")
        st.write("La región sombreada es el área entre la curva y el eje x. La parte sobre el eje suma y la parte bajo el eje resta.")
        datos = pd.DataFrame({"x": xs, "y": ys})
        area = alt.Chart(datos).mark_area(opacity=0.4).encode(x="x:Q", y="y:Q")
        linea = alt.Chart(datos).mark_line().encode(x="x:Q", y="y:Q")
        st.altair_chart(area + linea, width="stretch")
    except Exception:
        st.info("No se pudo hacer la verificación numérica para esta función.")

EJEMPLOS = [
    ("1. Derivada de un polinomio", "deriv", ["x^3 - 6*x^2 + 9*x + 1"]),
    ("2. Derivada de un producto", "deriv", ["x^2*sin(x)"]),
    ("3. Derivada de un cociente", "deriv", ["(3*x + 1)/(x^2 + 1)"]),
    ("4. Límite con forma 0/0", "lim", ["(x^2 - 9)/(x - 3)", "3"]),
    ("5. Límite trigonométrico", "lim", ["sin(x)/x", "0"]),
    ("6. Límite al infinito", "lim", ["(3*x^2 + 2*x)/(x^2 - 1)", "oo"]),
    ("7. Integral indefinida", "int", ["3*x^2 + 2*x + 1"]),
    ("8. Integral definida", "intdef", ["sin(x)", "0", "pi"]),
    ("9. Ecuación cuadrática", "ecu", ["x^2 - 5*x + 6"]),
    ("10. Factorización", "fact", ["x^3 - 6*x^2 + 11*x - 6"]),
    ("11. Dominio de una función", "dom", ["sqrt(x - 2)"]),
    ("12. Puntos críticos y extremos", "ext", ["x^3 - 6*x^2 + 9*x + 1"]),
]


def resolver_ejemplo(tipo, datos):
    f, _ = leer_funcion(datos[0])
    if tipo == "deriv":
        st.latex("f(x) = " + sp.latex(f))
        st.write("Aplicamos las reglas de derivación (suma, producto, cociente o cadena):")
        st.latex("f'(x) = " + sp.latex(sp.simplify(sp.diff(f, x))))
    elif tipo == "lim":
        a, _ = leer_funcion(datos[1])
        st.latex(r"\lim_{x \to " + sp.latex(a) + "} " + sp.latex(f))
        st.write("Valor del límite:")
        st.latex("= " + sp.latex(sp.limit(f, x, a)))
    elif tipo == "int":
        st.latex(r"\int " + sp.latex(f) + r"\,dx")
        st.write("Integramos término por término:")
        st.latex("= " + sp.latex(sp.integrate(f, x)) + " + C")
    elif tipo == "intdef":
        a, _ = leer_funcion(datos[1])
        b, _ = leer_funcion(datos[2])
        F = sp.integrate(f, x)
        st.latex(r"\int_{" + sp.latex(a) + "}^{" + sp.latex(b) + "} " + sp.latex(f) + r"\,dx")
        st.write("Buscamos la antiderivada F y calculamos F(b) − F(a):")
        st.latex("F(x) = " + sp.latex(F))
        st.latex("= " + sp.latex(sp.simplify(F.subs(x, b) - F.subs(x, a))))
    elif tipo == "ecu":
        st.latex(sp.latex(f) + " = 0")
        st.write("Soluciones:")
        for s in sp.solve(f, x):
            st.latex("x = " + sp.latex(s))
    elif tipo == "fact":
        st.latex(sp.latex(f) + " = " + sp.latex(sp.factor(f)))
    elif tipo == "dom":
        st.latex("f(x) = " + sp.latex(f))
        st.latex(r"\text{Dom}(f) = " + sp.latex(continuous_domain(f, x, sp.S.Reals)))
    elif tipo == "ext":
        d1 = sp.diff(f, x)
        d2 = sp.diff(d1, x)
        st.latex("f'(x) = " + sp.latex(d1) + r"\qquad f''(x) = " + sp.latex(d2))
        for c in sp.solve(d1, x):
            y = f.subs(x, c)
            v = d2.subs(x, c)
            if v > 0:
                clase = "mínimo relativo"
            elif v < 0:
                clase = "máximo relativo"
            else:
                clase = "el criterio no decide"
            st.write(f"x = {c}: f''({c}) = {v}, entonces hay un {clase} en el punto ({c}, {y})")
    st.caption("Para ver el procedimiento completo, usa el módulo correspondiente del menú.")


def generar_ejercicio(tipo):
    if tipo == "Derivada":
        n = random.randint(2, 5)
        a, b, c = random.randint(2, 7), random.randint(1, 9), random.randint(1, 9)
        f = a * x**n - b * x + c
        return {
            "tipo": tipo,
            "latex": "f(x) = " + sp.latex(f),
            "pregunta": "Calcula f'(x).",
            "formato": "Escribe con ^ y *, por ejemplo: 6*x^2 - 3",
            "respuesta": sp.diff(f, x),
            "pista": "Usa d/dx(xⁿ) = n·xⁿ⁻¹ en cada término. La derivada de una constante es 0.",
        }
    if tipo == "Integral indefinida":
        f = 3 * random.randint(1, 3) * x**2 + 2 * random.randint(1, 4) * x + random.randint(1, 9)
        return {
            "tipo": tipo,
            "f": f,
            "latex": r"\int " + sp.latex(f) + r"\,dx",
            "pregunta": "Calcula la integral indefinida.",
            "formato": "No escribas +C. Ejemplo: x^3 + x^2 + 5*x",
            "respuesta": sp.integrate(f, x),
            "pista": "Usa ∫xⁿ dx = xⁿ⁺¹/(n+1) en cada término. Puedes comprobar derivando tu resultado.",
        }
    if tipo == "Integral definida":
        f = 3 * random.randint(1, 3) * x**2 + random.randint(1, 5)
        k = random.randint(1, 4)
        return {
            "tipo": tipo,
            "latex": r"\int_{0}^{" + str(k) + "} (" + sp.latex(f) + r")\,dx",
            "pregunta": "Calcula la integral definida.",
            "formato": "Escribe un número, por ejemplo: 14",
            "respuesta": sp.integrate(f, (x, 0, k)),
            "pista": "Encuentra la antiderivada F(x) y calcula F(b) − F(a).",
        }
    if tipo == "Límite":
        k = random.randint(1, 6)
        f = (x**2 - k**2) / (x - k)
        return {
            "tipo": tipo,
            "latex": r"\lim_{x \to " + str(k) + "} " + sp.latex(f),
            "pregunta": "Calcula el límite.",
            "formato": "Escribe un número, por ejemplo: 4",
            "respuesta": sp.Integer(2 * k),
            "pista": "Al sustituir sale 0/0. Factoriza el numerador (diferencia de cuadrados) y cancela.",
        }
    if tipo == "Ecuación cuadrática":
        r1 = random.randint(-6, 6)
        r2 = random.randint(-6, 6)
        while r2 == r1:
            r2 = random.randint(-6, 6)
        f = sp.expand((x - r1) * (x - r2))
        return {
            "tipo": tipo,
            "latex": sp.latex(f) + " = 0",
            "pregunta": "Resuelve la ecuación.",
            "formato": "Escribe las dos soluciones separadas por coma, por ejemplo: 2, -3",
            "respuesta": [sp.Integer(r1), sp.Integer(r2)],
            "pista": "Usa la fórmula general o busca dos números que multiplicados den c y sumados den b.",
        }
    r1 = random.randint(1, 7)
    r2 = random.randint(1, 7)
    f = x**2 + (r1 + r2) * x + r1 * r2
    return {
        "tipo": tipo,
        "f": f,
        "latex": sp.latex(f),
        "pregunta": "Factoriza la expresión.",
        "formato": "Ejemplo: (x+2)*(x+3)",
        "respuesta": (x + r1) * (x + r2),
        "pista": "Busca dos números que multiplicados den el término independiente y sumados den el coeficiente de x.",
    }


def revisar_respuesta(ej, texto):
    """Devuelve (correcto, mensaje_de_error)."""
    tipo = ej["tipo"]
    try:
        if tipo == "Ecuación cuadrática":
            valores = set()
            for parte in texto.replace(";", ",").split(","):
                if parte.strip():
                    v, err = leer_funcion(parte)
                    if err:
                        return None, err
                    valores.add(sp.nsimplify(v))
            return valores == set(ej["respuesta"]), None
        resp, err = leer_funcion(texto)
        if err:
            return None, err
        if tipo == "Integral indefinida":
            return sp.simplify(sp.diff(resp, x) - ej["f"]) == 0, None
        if tipo == "Factorización":
            igual = sp.simplify(sp.expand(resp) - sp.expand(ej["f"])) == 0
            return igual and not resp.is_Add, None
        return sp.simplify(resp - ej["respuesta"]) == 0, None
    except Exception:
        return None, "No pude revisar tu respuesta. Revisa cómo la escribiste."


def mostrar_solucion(ej):
    if ej["tipo"] == "Ecuación cuadrática":
        for r in ej["respuesta"]:
            st.latex("x = " + sp.latex(r))
    elif ej["tipo"] == "Integral indefinida":
        st.latex(sp.latex(ej["respuesta"]) + " + C")
    else:
        st.latex(sp.latex(ej["respuesta"]))


def modulo_practica():
    st.header("Ejercicios y práctica")
    pestana1, pestana2 = st.tabs(["Ejemplos resueltos", "Generador de prácticas"])

    with pestana1:
        st.write("12 ejercicios de ejemplo. Elige uno y el asistente lo resuelve.")
        opcion = st.selectbox("Ejercicio:", [e[0] for e in EJEMPLOS], key="ej_sel")
        _, tipo_ej, datos = next(e for e in EJEMPLOS if e[0] == opcion)
        if st.button("Resolver ejemplo", key="ej_btn"):
            try:
                resolver_ejemplo(tipo_ej, datos)
            except Exception:
                st.error("No pude resolver este ejemplo.")

    with pestana2:
        st.write("Practica con ejercicios nuevos cada vez. Escribe tu respuesta y el asistente la revisa.")
        tipos = [
            "Derivada", "Integral indefinida", "Integral definida",
            "Límite", "Ecuación cuadrática", "Factorización",
        ]
        tipo = st.selectbox("Tipo de ejercicio:", tipos, key="prac_tipo")
        st.session_state.setdefault("prac_ok", 0)
        st.session_state.setdefault("prac_total", 0)

        nuevo = st.button("Nuevo ejercicio", key="prac_nuevo")
        if nuevo or "ejercicio" not in st.session_state or st.session_state["ejercicio"]["tipo"] != tipo:
            st.session_state["ejercicio"] = generar_ejercicio(tipo)
            st.session_state["prac_resp"] = ""
        ej = st.session_state["ejercicio"]

        st.latex(ej["latex"])
        st.write(ej["pregunta"])
        st.caption(ej["formato"])
        respuesta = st.text_input("Tu respuesta:", key="prac_resp")

        if st.button("Revisar respuesta", key="prac_rev"):
            if respuesta.strip() == "":
                st.warning("Escribe tu respuesta antes de revisarla.")
            else:
                correcto, error = revisar_respuesta(ej, respuesta)
                if error:
                    st.warning(error)
                else:
                    if not ej.get("contado"):
                        ej["contado"] = True
                        st.session_state["prac_total"] += 1
                        if correcto:
                            st.session_state["prac_ok"] += 1
                    if correcto:
                        st.success("¡Correcto! Muy bien.")
                    else:
                        st.error("Todavía no es correcto.")
                        st.info("Pista: " + ej["pista"])
                        with st.expander("Ver la solución"):
                            mostrar_solucion(ej)

        st.write(
            f"Puntaje: {st.session_state['prac_ok']} de {st.session_state['prac_total']} "
            "correctos al primer intento"
        )

def app_razones():
    st.write("La derivada es la **razón de cambio instantánea**. Usa x como variable (por ejemplo, x = tiempo).")
    contextos = {
        "Posición de un móvil": ("posición", "m", "s", "x^3 - 6*x^2 + 9*x", "velocidad", "aceleración"),
        "Costo de producción": ("costo", "quetzales", "unidades", "x^2/2 + 10*x + 200", "costo marginal", "cambio del costo marginal"),
        "Población": ("población", "personas", "años", "1000 + 50*x + 2*x^2", "velocidad de crecimiento", "aceleración del crecimiento"),
        "Otro contexto": ("cantidad", "u", "t", "x^2 + 3*x", "razón de cambio", "segunda razón de cambio"),
    }
    contexto = st.selectbox("Contexto:", list(contextos.keys()), key="raz_ctx")
    nombre, u_f, u_x, ejemplo, nom1, nom2 = contextos[contexto]

    col1, col2 = st.columns(2)
    u_f = col1.text_input("Unidad de la cantidad:", value=u_f, key="raz_uf_" + contexto)
    u_x = col2.text_input("Unidad de x:", value=u_x, key="raz_ux_" + contexto)
    texto = st.text_input(f"{nombre} f(x) =", value=ejemplo, key="raz_f_" + contexto)
    c1, c2, c3 = st.columns(3)
    x0 = c1.number_input("Punto x₀:", value=2.0, key="raz_x0")
    a = c2.number_input("Intervalo desde a:", value=0.0, key="raz_a")
    b = c3.number_input("hasta b:", value=4.0, key="raz_b")

    f, error = leer_funcion(texto)
    if error:
        st.warning(error)
        return
    if a >= b:
        st.warning("En el intervalo, a debe ser menor que b.")
        return
    if not st.button("Calcular razones de cambio", key="raz_btn"):
        return

    x0, a, b = sp.Rational(str(x0)), sp.Rational(str(a)), sp.Rational(str(b))
    try:
        fa, fb, f0 = f.subs(x, a), f.subs(x, b), f.subs(x, x0)
        if not all(v.is_real and v.is_finite for v in (fa, fb, f0)):
            raise ValueError
    except Exception:
        st.error("La función no está definida en alguno de los puntos elegidos. Cambia a, b o x₀.")
        return

    st.subheader("1. Función")
    st.latex("f(x) = " + sp.latex(f))
    st.write(f"f(x) representa la {nombre} (en {u_f}) y x está en {u_x}.")

    st.subheader("2. Razón de cambio promedio en [a, b]")
    st.latex(r"\frac{f(b) - f(a)}{b - a}")
    st.latex(
        r"= \frac{f(" + sp.latex(b) + ") - f(" + sp.latex(a) + ")}{" + sp.latex(b) + " - " + sp.latex(a)
        + r"} = \frac{" + sp.latex(fb) + " - (" + sp.latex(fa) + ")}{" + sp.latex(b - a) + "}"
    )
    promedio = sp.simplify((fb - fa) / (b - a))
    st.latex("= " + sp.latex(promedio))
    st.write(f"En promedio, la {nombre} cambia {float(promedio):.4f} {u_f} por cada {u_x}.")

    st.subheader("3. Razón de cambio instantánea en x₀")
    d1 = sp.simplify(sp.diff(f, x))
    st.write(f"La {nom1} es la derivada f'(x):")
    st.latex("f'(x) = " + sp.latex(d1))
    v1 = sp.simplify(d1.subs(x, x0))
    st.latex("f'(" + sp.latex(x0) + ") = " + sp.latex(v1))
    valor = float(v1)
    if valor > 0:
        st.success(f"En x = {float(x0)} la {nombre} está aumentando a razón de {valor:.4f} {u_f} por {u_x}.")
    elif valor < 0:
        st.success(f"En x = {float(x0)} la {nombre} está disminuyendo a razón de {abs(valor):.4f} {u_f} por {u_x}.")
    else:
        st.success(f"En x = {float(x0)} la {nombre} no está cambiando en ese instante (f' = 0).")

    st.subheader("4. Segunda derivada")
    d2 = sp.simplify(sp.diff(d1, x))
    st.write(f"La {nom2} es la segunda derivada f''(x):")
    st.latex("f''(x) = " + sp.latex(d2))
    v2 = sp.simplify(d2.subs(x, x0))
    st.latex("f''(" + sp.latex(x0) + ") = " + sp.latex(v2))
    if float(v2) > 0:
        st.write("f'' > 0: la razón de cambio está aumentando.")
    elif float(v2) < 0:
        st.write("f'' < 0: la razón de cambio está disminuyendo.")
    else:
        st.write("f'' = 0: la razón de cambio no varía en ese instante.")


def optimizar_contexto(f, lo, hi, que_es_x, que_es_f, buscar_max):
    st.subheader("2. Función a optimizar")
    st.latex("f(x) = " + sp.latex(sp.expand(f)))
    st.write(f"Aquí x es {que_es_x} y f(x) es {que_es_f}. Dominio: de {sp.N(lo, 6)} a {sp.N(hi, 6)}.")

    d1 = sp.diff(f, x)
    d2 = sp.diff(d1, x)
    st.subheader("3. Puntos críticos")
    st.write("Derivamos e igualamos a cero:")
    st.latex("f'(x) = " + sp.latex(sp.simplify(d1)) + " = 0")
    try:
        soluciones = sp.solve(sp.Eq(d1, 0), x)
    except Exception:
        st.error("No pude resolver f'(x) = 0.")
        return None
    criticos = [c for c in soluciones if c.is_real and float(lo) <= float(c) <= float(hi)]
    if not criticos:
        st.warning("No hay puntos críticos dentro del dominio.")
        return None
    for c in criticos:
        st.latex("x = " + sp.latex(c))

    st.subheader("4. Comparamos valores")
    st.write("Evaluamos f en los puntos críticos y en los extremos del dominio:")
    filas = [(c, sp.simplify(f.subs(x, c))) for c in criticos + [lo, hi]]
    st.table(pd.DataFrame({
        "x": [str(sp.N(c, 6)) for c, _ in filas],
        "f(x)": [str(sp.N(y, 6)) for _, y in filas],
    }))
    mejor = (max if buscar_max else min)(filas, key=lambda t: float(t[1]))

    st.subheader("5. Verificación con la segunda derivada")
    st.latex("f''(x) = " + sp.latex(sp.simplify(d2)))
    try:
        v = sp.simplify(d2.subs(x, mejor[0]))
        st.latex("f''(" + sp.latex(mejor[0]) + ") = " + sp.latex(v))
        if v < 0:
            st.write("f'' < 0: la curva es cóncava hacia abajo, se confirma un máximo.")
        elif v > 0:
            st.write("f'' > 0: la curva es cóncava hacia arriba, se confirma un mínimo.")
    except Exception:
        pass
    return mejor


def app_optimizacion():
    st.write("Problemas de la vida real: planteamos la función, derivamos y buscamos el máximo o el mínimo.")
    problema = st.selectbox(
        "Problema:",
        ["Cerca rectangular (área máxima)", "Caja sin tapa (volumen máximo)", "Proyectil (altura máxima)"],
        key="opt_prob",
    )

    if problema.startswith("Cerca"):
        P = st.number_input("Metros de cerca disponibles:", min_value=1.0, value=100.0, key="opt_P")
        st.write("Un agricultor tiene P metros de cerca y quiere cercar un terreno rectangular con el **área máxima**.")
        if not st.button("Resolver problema", key="opt_b1"):
            return
        Pq = sp.Rational(str(P))
        st.subheader("1. Planteamiento")
        st.write("Sea x el ancho y y el largo. El perímetro es 2x + 2y = P, así que:")
        st.latex("y = " + sp.latex(Pq / 2) + " - x")
        st.latex("A = x \\cdot y = x(" + sp.latex(Pq / 2) + " - x)")
        mejor = optimizar_contexto(x * (Pq / 2 - x), sp.Integer(0), Pq / 2, "el ancho (m)", "el área (m²)", True)
        if mejor:
            st.subheader("6. Respuesta")
            st.success(
                f"El terreno de área máxima mide {sp.N(mejor[0], 6)} m de ancho y "
                f"{sp.N(Pq / 2 - mejor[0], 6)} m de largo. Su área es {sp.N(mejor[1], 6)} m²."
            )

    elif problema.startswith("Caja"):
        L = st.number_input("Lado del cartón cuadrado (cm):", min_value=1.0, value=30.0, key="opt_L")
        st.write("De un cartón cuadrado de lado L se cortan cuadrados de lado x en las esquinas y se doblan los lados. Queremos el **volumen máximo**.")
        if not st.button("Resolver problema", key="opt_b2"):
            return
        Lq = sp.Rational(str(L))
        st.subheader("1. Planteamiento")
        st.write("La base mide (L − 2x) por (L − 2x) y la altura es x:")
        st.latex("V(x) = x(" + sp.latex(Lq) + " - 2x)^2")
        mejor = optimizar_contexto(x * (Lq - 2 * x) ** 2, sp.Integer(0), Lq / 2, "el lado del cuadrado cortado (cm)", "el volumen (cm³)", True)
        if mejor:
            st.subheader("6. Respuesta")
            st.success(
                f"Se deben cortar cuadrados de {sp.N(mejor[0], 6)} cm de lado. "
                f"El volumen máximo es {sp.N(mejor[1], 6)} cm³."
            )

    else:
        c1, c2 = st.columns(2)
        h0 = c1.number_input("Altura inicial (m):", min_value=0.0, value=0.0, key="opt_h0")
        v0 = c2.number_input("Velocidad inicial hacia arriba (m/s):", min_value=0.1, value=20.0, key="opt_v0")
        st.write("Se lanza un proyectil hacia arriba. Su altura es h(t) = h₀ + v₀t − 4.9t² (aquí x = tiempo en segundos).")
        if not st.button("Resolver problema", key="opt_b3"):
            return
        h0q, v0q = sp.Rational(str(h0)), sp.Rational(str(v0))
        h = h0q + v0q * x - sp.Rational(49, 10) * x**2
        st.subheader("1. Planteamiento")
        st.latex("h(t) = " + sp.latex(sp.expand(h)))
        raices = [r for r in sp.solve(h, x) if r.is_real and r >= 0]
        if not raices:
            st.error("El proyectil nunca llega al suelo con estos datos.")
            return
        t_suelo = max(raices, key=lambda r: float(r))
        st.write("El proyectil toca el suelo cuando h(t) = 0:")
        st.latex("t = " + sp.latex(sp.simplify(t_suelo)) + r" \approx " + sp.latex(sp.N(t_suelo, 4)) + r"\ \text{s}")
        mejor = optimizar_contexto(h, sp.Integer(0), t_suelo, "el tiempo (s)", "la altura (m)", True)
        if mejor:
            st.subheader("6. Respuesta")
            st.success(
                f"La altura máxima es {sp.N(mejor[1], 6)} m y se alcanza a los {sp.N(mejor[0], 6)} s. "
                f"El proyectil cae al suelo a los {sp.N(t_suelo, 6)} s."
            )


def modulo_aplicaciones():
    st.header("Aplicaciones")
    opcion = st.radio(
        "Tipo de problema:",
        ["Extremos de una función", "Razones de cambio", "Optimización con contexto"],
        horizontal=True,
        key="apl_op",
    )
    if opcion == "Extremos de una función":
        app_extremos()
    elif opcion == "Razones de cambio":
        app_razones()
    else:
        app_optimizacion()       
# ---------- Página principal ----------
st.title("CalcUMG")
st.caption("Asistente Virtual de Cálculo y Precálculo")
st.caption("Universidad Mariano Gálvez · Sección A · Integrantes: José Guillermo Chavez Suhul y Chelsy Adely Martinez Barrios · Asignatura: precalculo")

modulo = st.sidebar.radio(
    "Módulos",
    [
        "Álgebra y funciones",
        "Gráficas",
        "Límites",
        "Derivadas",
        "Integrales",
        "Aplicaciones",
        "Ejercicios y práctica",
    ],
    index=3,
)

if modulo == "Derivadas":
    modulo_derivadas()
elif modulo == "Aplicaciones":
    modulo_aplicaciones()
elif modulo == "Gráficas":
    modulo_graficas()
elif modulo == "Álgebra y funciones":
    modulo_algebra()
elif modulo == "Límites":
    modulo_limites()
elif modulo == "Integrales":
    modulo_integrales()
elif modulo == "Ejercicios y práctica":
    modulo_practica()    
else:
    st.info("Este módulo está en construcción.")