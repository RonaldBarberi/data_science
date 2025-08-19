"""
create_at: 2025-08-18 19:07

update_at: 2025-08-18 19:07

@author: Ronal.Barberi
"""

#%% Imported libraries

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% Create class

class StatisticalREBR:

    @staticmethod
    def mode(col_quantitative):
        freq = col_quantitative.value_counts()
        Mo = freq.index[0]

        return Mo


    @staticmethod
    def median(col_quantitative):
        N = len(col_quantitative)
        dt_order_asc = col_quantitative.sort_values().reset_index(drop=True)

        if N % 2 == 1:
            Me = dt_order_asc[N // 2]
        else:
            Me = (dt_order_asc[(N // 2) - 1] + dt_order_asc[N // 2]) / 2

        return round(Me, 2)


    @staticmethod
    def arithmetic_mean(col_quantitative):
        E_xi = col_quantitative.sum()
        N = len(col_quantitative)
        u = round(E_xi / N, 2)

        return u


    @staticmethod
    def variance(col_quantitative, type_dt: str):
        N = len(col_quantitative)
        u = StatisticalREBR.arithmetic_mean(col_quantitative)

        if type_dt == 'P': # sigma2
            var = round(np.sum((col_quantitative - u) ** 2) / N, 2)

        elif type_dt == 'M': # s2
            var = round(np.sum((col_quantitative - u) ** 2) / (N - 1), 2)
        
        else:
            print(f'Type variable not is correct: {type_dt}')

        return var

        
    @staticmethod
    def deviation(col_quantitative, type_dt: str):
        sigma2_s2 = StatisticalREBR.variance(col_quantitative, type_dt)
        dev = round(np.sqrt(sigma2_s2), 2)

        return dev
   
   
    @staticmethod
    def coefficient_variation(col_quantitative, type_dt: str, print_result=False):
        """
        - CV <= 0.10 : baja dispersión
        - 0.10 < CV <= 0.30 : dispersión moderada
        - CV > 0.30 : alta dispersión (usar mediana como tendencia central)
        """
        sigma2_s2 = StatisticalREBR.deviation(col_quantitative, type_dt)
        u = StatisticalREBR.arithmetic_mean(col_quantitative)
        cv = round((sigma2_s2 / u) * 100, 2)

        if print_result is True:
            print(f'The coefficient variation is: {cv:.2f}')

        return cv


    @staticmethod
    def asymmetry_coefficient(col_quantitative, type_dt: str, graph=False):
        """
        - As > 0 : distribución asimétrica positiva o sesgada a la derecha
        - As < 0 : distribución asimétrica negativa o sesgada a la izquierda
        - As = 0 : los datos siguen una distribución simétrica.
        """
        x = np.asarray(col_quantitative, dtype=float)
        n = len(x)
        dev = StatisticalREBR.deviation(x, type_dt)
        u = StatisticalREBR.arithmetic_mean(col_quantitative)
        Me = StatisticalREBR.median(col_quantitative)

        if dev == 0:
            return 0.0

        m3 = np.sum((x - u) ** 3) / n
        g1 = m3 / (dev ** 3)

        if type_dt.upper() == 'P':
            As = g1
        elif type_dt.upper() == 'M':
            if n < 3:
                raise ValueError("n debe ser >= 3 para usar el ajuste muestral.")
            As = (np.sqrt(n * (n - 1)) / (n - 2)) * g1
        else:
            raise ValueError("type_dt debe ser 'P' o 'M'")

        As = round(float(As), 4)
        
        if graph is True:
            fig, ax = plt.subplots(figsize=(14, 4), dpi=120)

            sns.kdeplot(col_quantitative, ax=ax, fill=False, color='#00aea9', linewidth=2)

            ax.axvline(u,  color='red',     linestyle='--', label=f'Arithmetic Mean = {u:.2f}')
            ax.axvline(Me, color='#00FF00', linestyle='-.', label=f'Median = {Me:.2f}')

            # ax.set_title(f'Gráfica de asimetría {col_quantitative.capitalize()}')
            name = getattr(col_quantitative, "name", None) or "variable"
            nice = str(name).replace("_", " ").title()   # opcional: formatea
            ax.set_title(f"Gráfica de asimetría — {nice}")
            ax.set_facecolor('#5A5A59')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='white')
            ax.set_axisbelow(True)
            ax.legend()
            plt.show()
        
        elif graph is False:
            pass

        else:
            raise ValueError('graph debe ser True o False')

        print(f'The asymmetry coefficient is: {As:.2f}')
        return As
    

    @staticmethod
    def percentile(col_quantitative, K: int):
        """
        K = Porcentaje de interés.
        """
        sorted_data = sorted(col_quantitative)
        N = len(sorted_data)
        pos = (N - 1) * K / 100
        lower = int(np.floor(pos))
        upper = int(np.ceil(pos))
        
        if lower == upper:
            i = sorted_data[int(pos)]
            
        else:
            i = sorted_data[lower] + (pos - lower) * (sorted_data[upper] - sorted_data[lower])

        return i


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
