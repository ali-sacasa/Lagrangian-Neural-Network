# Redes Neuronales Lagrangianas (LNN) — Proyecto RNL

Implementación en JAX de una **Red Neuronal Lagrangiana** para aprender la
dinámica de sistemas físicos directamente desde trayectorias, sin imponer
explícitamente las ecuaciones de movimiento. La red aprende un Lagrangiano
escalar `L_θ(q, q̇)` y la dinámica se obtiene aplicando automáticamente la
ecuación de Euler–Lagrange vía diferenciación automática.

Como sistemas de prueba se incluyen:

1. Partícula cargada bajo fuerza de Lorentz en 2D (campos `E` y `B` constantes).
2. Péndulo doble.

Los datos de entrenamiento se generan mediante un integrador estocástico
(Euler–Maruyama) de una SDE de Langevin con fricción, con filtros de
regularidad sobre la energía conservada `J(q,v) = v·∂L/∂v − L` y sobre
`‖dL‖` para asegurar condiciones iniciales suaves.

## Autores

- **Juan Manuel Bonilla Arce** — `juan.bonillaarce@ucr.ac.cr`
- **Sebastián Alí Sacasa Céspedes** — `sebastian.sacasa@ucr.ac.cr`

Escuela de Física, Universidad de Costa Rica (UCR),
San Pedro de Montes de Oca, San José, 11501-2060, Costa Rica.


## Requisitos

- Python ≥ 3.10
- `jax`, `jaxlib`
- `optax`
- `numpy`, `matplotlib`

Instalación rápida:

```bash
pip install -U jax jaxlib optax numpy matplotlib
```
Para GPU/TPU consultar la guía oficial de instalación de JAX
(https://github.com/jax-ml/jax#installation).

Uso
Abrir ProyectoRNL.ipynb en Jupyter / VS Code y ejecutar las celdas
en orden. El notebook está dividido en:

Definición del Lagrangiano objetivo (L_true).
Ecuación de Euler–Lagrange e integrador determinista de referencia.
Generación de datos vía SDE de Langevin con filtros de regularidad.
Definición de la red L_θ (MLP con activación softplus para
garantizar regularidad C²) y la función de pérdida sobre el campo
vectorial inducido.
Bucle de entrenamiento con optax (Adam + clip de gradiente).
Diagnóstico: curvas de pérdida, comparación de trayectorias
L_true vs L_θ y conservación de la energía.

# Citación

Si usa este código, por favor cite:

@misc{bonilla_sacasa_2026_lnn,
  author       = {Bonilla Arce, Juan Manuel and Sacasa Céspedes, Sebastián Alí},
  title        = {Redes Neuronales Lagrangianas: implementación en JAX
                  con generación de datos vía SDE de Langevin},
  year         = {2026},
  institution  = {Universidad de Costa Rica, Escuela de Física},
  howpublished = {\url{https://github.com/<usuario>/<repositorio>}}
}

y el trabajo original de LNN.

Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., Ho, S.
Lagrangian Neural Networks. arXiv:2003.04630 (2020).

Licencia
Distribuido bajo licencia MIT. Véase el archivo LICENSE.

# © 2026 Juan Manuel Bonilla Arce y Sebastián Alí Sacasa Céspedes.

Referencias principales
Cranmer M., Greydanus S., Hoyer S., Battaglia P., Spergel D., Ho S.
Lagrangian neural networks. arXiv:2003.04630, 2020.
Mann P. Lagrangian and Hamiltonian Dynamics. Oxford University Press, 2018.
Greydanus S., Dzamba M., Yosinski J. Hamiltonian Neural Networks.
NeurIPS, 2019.
(La lista completa de referencias está al final del notebook.)



---

### Archivo `LICENSE` (MIT)

MIT License

Copyright (c) 2026 Juan Manuel Bonilla Arce, Sebastián Alí Sacasa Céspedes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Estructura
