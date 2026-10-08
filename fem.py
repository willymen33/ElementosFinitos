%%writefile fem.py

import numpy as np

def resolver(d1, d2, L, E, P, n):

    h = L / n

    x_nodos = np.linspace(
        0.0,
        L,
        n + 1
    )

    K = np.zeros(
        (n + 1, n + 1)
    )

    F = np.zeros(n + 1)

    for e in range(n):

        xm = (
            x_nodos[e] +
            x_nodos[e + 1]
        ) / 2.0

        dm = (
            d1 +
            (d2 - d1) * xm / L
        )

        A = np.pi * dm**2 / 4.0

        ke = (
            E * A / h
        ) * np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])

        K[e:e+2, e:e+2] += ke

    F[-1] = P

    u_nodos = np.zeros(n + 1)

    K_reducida = K[1:, 1:]
    F_reducido = F[1:]

    u_nodos[1:] = np.linalg.solve(
        K_reducida,
        F_reducido
    )

    sigma_elem = np.zeros(n)

    for e in range(n):

        sigma_elem[e] = (
            E *
            (
                u_nodos[e + 1] -
                u_nodos[e]
            ) / h
        )

    return (
        x_nodos,
        u_nodos,
        sigma_elem
    )
