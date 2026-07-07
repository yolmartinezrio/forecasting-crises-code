# Sistemas de Alerta Temprana en Crisis Financieras Sistémicas
### Integración de VAE Temporal y DDPM con Clasificadores Econométricos de Panel

Este repositorio contiene el código fuente y los desarrollos prácticos del **Trabajo Fin de Máster (TFM)** para el **Máster Universitario en Visual Analytics & Big Data**.

* **Autor:** Yolanda Martínez Río
* **Director:** Jesús Cigales Canga
* **Institución:** Universidad Internacional de La Rioja (UNIR)

---

## Resumen del Proyecto

El objetivo principal de este trabajo es el desarrollo de un **Sistema de Alerta Temprana (EWS - Early Warning System)** avanzado para la predicción de crisis financieras sistémicas. Debido a la naturaleza desbalanceada y compleja de los datos macroeconómicos y financieros históricos, este proyecto propone un enfoque híbrido innovador que combina el aprendizaje profundo generativo con la econometría tradicional de datos de panel:

1. **Codificadores Variacionales Autoasociativos Temporales (Temporal VAE):** Utilizados para la extracción de características dinámicas y la reducción de la dimensionalidad de las series temporales macrofinancieras.
2. **Modelos Probabilísticos de Difusión de Denoisificación (DDPM):** Empleados para la generación de datos sintéticos y el balanceo de clases (eventos de crisis frente a periodos de estabilidad).
3. **Clasificadores Econométricos de Panel:** Modelos logit/probit de efectos fijos o aleatorios entrenados sobre el espacio latente y los datos aumentados para estimar la probabilidad de ocurrencia de crisis sistémicas con alta interpretabilidad.
