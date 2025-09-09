"""
create_at: 2025-08-18 19:07

update_at: 2025-09-09 10:07

@author: Ronal.Barberi
"""

#%% Imported libraries

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% Create class

class StatisticalREBR:

    @staticmethod
    def limpieza_valores(values):
        arr = np.array(values, dtype="object")

        cleaned = []
        for v in arr:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                cleaned.append(fv)

        return cleaned


    @staticmethod
    def valor_min_max(col_quantitative):
        datos = StatisticalREBR.limpieza_valores(col_quantitative)
        val_min = min(datos)
        val_max = max(datos)

        return val_min, val_max


    @staticmethod
    def moda(col_quantitative):
        datos = StatisticalREBR.limpieza_valores(col_quantitative)

        if not datos:
            return None

        conteo = {}
        for v in datos:
            conteo[v] = conteo.get(v, 0) + 1

        Mo = max(conteo, key=conteo.get)

        return Mo


    @staticmethod
    def media(col_quantitative):
        datos = StatisticalREBR.limpieza_valores(col_quantitative)
        E_xi = sum(datos)
        N = len(datos)
        u = round(E_xi / N, 2)

        return u


    @staticmethod
    def mediana(col_quantitative):
        datos = StatisticalREBR.limpieza_valores(col_quantitative)
        N = len(datos)
        dt_order_asc = sorted(datos)

        if N % 2 == 1:
            Me = dt_order_asc[N // 2]
        else:
            Me = (dt_order_asc[(N // 2) - 1] + dt_order_asc[N // 2]) / 2

        return Me


    @staticmethod
    def varianza(col_quantitative, type_dt: str):
        datos = StatisticalREBR.limpieza_valores(col_quantitative)
        N = len(datos)
        u = StatisticalREBR.media(col_quantitative)
        sc = sum((x - u) ** 2 for x in datos) # usar suma de cuadrados de desviaciones

        if type_dt == 'P': # sigma2
            var = round(sc / N, 2)

        elif type_dt == 'M': # s2
            if N < 2:
                return None
            var = round(sc / (N - 1), 2)

        else:
            print(f'[ERROR] tipo de datos (poblacional/muestral) no aclarado: {type_dt}')

        return var

        
    @staticmethod
    def desviacion_estandar(col_quantitative, type_dt: str):
        sigma2_s2 = StatisticalREBR.varianza(col_quantitative, type_dt)
        dev_est = round(sigma2_s2** 0.5, 2)

        return dev_est
   
   
    @staticmethod
    def coeficiente_variacion(col_quantitative, type_dt: str, print_result=False):
        """
        - CV <= 0.10 : baja dispersión
        - 0.10 < CV <= 0.30 : dispersión moderada
        - CV > 0.30 : alta dispersión (usar mediana como tendencia central)
        """
        sigma2_s2 = StatisticalREBR.desviacion_estandar(col_quantitative, type_dt)
        u = StatisticalREBR.media(col_quantitative)
        cv = round((sigma2_s2 / abs(u)) * 100, 2)

        if abs(u) < 1e-6:
            print('[WARN] la media es demasiado cercana a 0, el CV no es interpretable.')

        if print_result is True:
            print(f'[OK] el coeficiente de variacion es: {cv:.2f}')
        
        return cv


    @staticmethod
    def coeficiente_asimetria(col_quantitative, type_dt: str, graph=False):
        """
        - As > 0 : distribución asimétrica positiva o sesgada a la derecha
        - As < 0 : distribución asimétrica negativa o sesgada a la izquierda
        - As = 0 : los datos siguen una distribución simétrica.
        """
        x_raw = np.asarray(col_quantitative, dtype=float)
        mask = np.isfinite(x_raw) # True si es número finito (no NaN/Inf)
        x = x_raw[mask]
        n = x.size
        dev = StatisticalREBR.desviacion_estandar(col_quantitative, type_dt)
        u = StatisticalREBR.media(col_quantitative)
        Me = StatisticalREBR.mediana(col_quantitative)

        if dev == 0:
            return 0.0

        m3 = np.sum((x - u) ** 3) / n
        g1 = m3 / (dev ** 3)

        if type_dt.upper() == 'P':
            As = g1
        elif type_dt.upper() == 'M':
            if n < 3:
                raise ValueError("[ERROR] n debe ser >= 3 para usar el ajuste muestral.")
            As = (np.sqrt(n * (n - 1)) / (n - 2)) * g1
        else:
            raise ValueError("[ERROR] type_dt debe ser 'P' o 'M'")

        As = round(float(As), 4)
        
        if graph is True:
            fig, ax = plt.subplots(figsize=(14, 4), dpi=120)

            sns.kdeplot(col_quantitative, ax=ax, fill=False, color='#00aea9', linewidth=2)
            ax.axvline(u,  color='red',     linestyle='--', label=f'Media = {u:.2f}')
            ax.axvline(Me, color='#00FF00', linestyle='-.', label=f'Mediana = {Me:.2f}')

            name = getattr(col_quantitative, 'name', None) or 'variable'
            nice = str(name).replace('_', ' ').title()   # opcional: formatea
            ax.set_title(f'Gráfica de asimetría — {nice}')
            ax.set_facecolor('#5A5A59')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='white')
            ax.set_axisbelow(True)
            ax.legend()
            plt.show()
        
        elif graph is False:
            pass

        else:
            raise ValueError('[ERROR] graph debe ser True o False')

        print(f'[OK] el coeficiente de asimetria es: {As:.2f}')
        return As
    

    @staticmethod
    def percentil(col_quantitative, K: int):
        """
        K = Porcentaje de interés.
        """
        datos = StatisticalREBR.limpieza_valores(col_quantitative)
        sorted_data = sorted(datos)
        N = len(sorted_data)
        P = (N - 1) * K / 100
        i = int(P)
        f = P - i
        
        if i + 1 < N:
            return sorted_data[i] + f * (sorted_data[i + 1] - sorted_data[i])
        else:
            return sorted_data[i]  # cuando P apunta al último elemento


    @staticmethod
    def grafic_multi_elements_barplot(fig_value, ax_val_x, ax_val_y, nam_df, lis_cols):
        fig, axes = plt.subplots(ax_val_x, ax_val_y, figsize=fig_value)
        axes = axes.flatten() if ax_val_x > 1 and ax_val_y > 1 else axes

        for i, col in enumerate(lis_cols):
            col_counts = nam_df[col].value_counts()
            sns.barplot(
                x=col_counts.index,
                y=col_counts.values,
                color='#00aea9',
                ax=axes[i]
            )
            axes[i].set_facecolor("#5A5A59")
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel("Amount regsters", fontsize=12)
            axes[i].set_title(f"Amount regsters to {col.capitalize()}", fontsize=14)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color='white')

        plt.tight_layout()
        plt.show()
