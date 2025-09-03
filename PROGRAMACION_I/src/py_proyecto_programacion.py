#%% Importar librerias

import os
import numpy as np
import pandas as pd

#%% Iniciar el proceso

ruta_actual = os.path.abspath(os.path.dirname(__file__))
path_bbdd = os.path.join(ruta_actual, '..', 'data', 'bbdd_estudiantes.csv')
cols = [
    'id',
    'documento_estudiante',
    'nombre_estudiante',
    'cantidad_hermanos',
    'nota_programacion',
    'nota_estadistica',
    'nota_matematicas'
]

pd.set_option('display.max_rows', None)       # mostrar todas las filas
pd.set_option('display.max_columns', None)   # mostrar todas las columnas
pd.set_option('display.width', 1000)         # ancho máximo de línea
pd.set_option('display.colheader_justify', 'center')  # centrar nombres de columnas
df_bbdd = pd.read_csv(path_bbdd, sep=',', encoding='latin-1', usecols=cols)
max_id = df_bbdd['id'].max()

while True:
    insert_nuevo_reg = input('¿Deseas agregar un nuevo estudiante a la base de datos? (S/N): ')
    if insert_nuevo_reg.upper() == 'S':
        datos = []
        documento_estudiante_inpt = input('Escribe el documento del estudiante (solamente numeros enteros): ')
        nombre_estudiante_inpt = input('Escribe el nombre del estudiante: ')
        cantidad_hermanos_inpt = input('Escribe la cantidad de hermanos (numero entero. Ejem: 1): ')
        nota_programacion_inpt = input('Escribe la nota para la materia de programacion (valor con decimal. Ejem: 1.0): ')
        nota_estadistica_inpt = input('Escribe la nota para la materia de estadistica (valor con decimal. Ejem: 1.0): ')
        nota_matematicas_inpt = input('Escribe la nota para la materia de matematicas (valor con decimal. Ejem: 1.0): ')

        cols_notas = [nota_programacion_inpt, nota_estadistica_inpt, nota_matematicas_inpt]
        notas_validas = True
        for nota in cols_notas:
            valor = float(nota)
            if not (0.0 <= valor <= 5.0):
                print(f"[ERROR] La nota {nota} no es válida. Debe estar entre 0.0 y 5.0 (Volviendo al inicio)")
                notas_validas = False

        if not notas_validas:
            continue

        nuevos_datos = pd.DataFrame([{
            'id': max_id + 1,
            'documento_estudiante': int(documento_estudiante_inpt),
            'nombre_estudiante': nombre_estudiante_inpt.upper(),
            'cantidad_hermanos': int(cantidad_hermanos_inpt),
            'nota_programacion': float(nota_programacion_inpt),
            'nota_estadistica': float(nota_estadistica_inpt),
            'nota_matematicas': float(nota_matematicas_inpt)
        }])
        print(f'[INFO] Se insertara el nuevo registro: {nuevos_datos}')
        try:
            df_bbdd = pd.concat([df_bbdd, nuevos_datos], ignore_index=True)
            df_bbdd.to_csv(path_bbdd, index=False, sep=',')
            print('[OK] Nuevo registro alamcenado correctamente.')
            break
        except Exception as err:
            print(f'[ERROR] No se puedo guardar el nuevo registro, error explicito: {err}')

    elif insert_nuevo_reg.upper() == 'N':
        break

    else:
        print('Opción incorrecta, por favor selecciona "S" para si o "N" para no.')

#%% Validaciones

while True:

    opcion_usuario = input('¿Cual de las siguientes alternativas quieres realizar? \n' \
    '1. Visualizar los datos de uno o mas estudiantes. \n' \
    '2. Calular el promedio de notas de uno o mas estudiantes. \n' \
    '3. Calcular descuento en la matricula por cantidad de hermanos matriculados de uno o mas estudiantes. \n' \
    '4. Calcular cantidad de estudiantes aprobados y reprobados. \n' \
    '5. Calcular el dinero total recaudado por matriculas. \n' \
    '6. Calcular el promedio general del colegio. \n' \
    '7. Calcular el rendimiendo academico. \n' \
    '8. Calcular el dinero ahorrado en descuentos. \n' \
    'Por favor indicanos el numero de la opción que quieres consultar: ')

    opcion_valida = True
    num_opcion = int(opcion_usuario)
    if not (1 <= num_opcion <= 8):
        print(f'[ERROR] La opcion digitada "{num_opcion}" no es válida. Debe estar entre 1 y 8 (Volviendo al inicio) \n \
              ')
        opcion_valida = False

    if not opcion_valida:
        continue

    print(f'[INFO] Opcion elegida: {opcion_usuario}. Procediendo con la tarea.')
    
    if num_opcion == 1:
        cc_estudiantes = input('Digita el documentos que deseas validar (En caso de ser varios, dividirlos por ","): ')

        ids = [int(x) for x in cc_estudiantes.split(',')]
        df_filtrado = df_bbdd.query('documento_estudiante in @ids')

        print(df_filtrado.to_markdown(index=False))
    
    elif num_opcion == 2:
        cc_estudiantes = input('Digita el documentos que deseas validar (En caso de ser varios, dividirlos por ","): ')

        ids = [int(x) for x in cc_estudiantes.split(',')]
        df_filtrado = df_bbdd.query('documento_estudiante in @ids')
        df_filtrado['promedio_notas'] = df_filtrado[['nota_programacion', 'nota_estadistica', 'nota_matematicas']].mean(axis=1).round(2)
        df_filtrado = df_filtrado[['id', 'documento_estudiante', 'nombre_estudiante', 'nota_programacion', 'nota_estadistica', 'nota_matematicas', 'promedio_notas']]

        print(df_filtrado.to_markdown(index=False))
        
    elif num_opcion == 3:
        cc_estudiantes = input('Digita el documentos que deseas validar (En caso de ser varios, dividirlos por ","): ')

        ids = [int(x) for x in cc_estudiantes.split(',')]
        df_filtrado = df_bbdd.query('documento_estudiante in @ids')
        costo_matricula = 200_000
        condiciones = [
            df_filtrado['cantidad_hermanos'] >= 3,
            df_filtrado['cantidad_hermanos'] == 2,
            df_filtrado['cantidad_hermanos'] == 1
        ]
        valores = [0.20, 0.15, 0.10]

        df_filtrado['descuento_pct'] = np.select(condiciones, valores, default=0.0)

        df_filtrado['descuento_valor'] = costo_matricula * df_filtrado['descuento_pct']
        df_filtrado['costo_matricula'] = costo_matricula - df_filtrado['descuento_valor']

        df_filtrado = df_filtrado[['id','documento_estudiante','nombre_estudiante','cantidad_hermanos','descuento_pct','descuento_valor','costo_matricula']].copy()
        df_filtrado['descuento_pct'] = (df_filtrado['descuento_pct']*100).round(0).astype(int).astype(str) + '%'
        df_filtrado['descuento_valor'] = df_filtrado['descuento_valor'].round(0).map('{:,.0f}'.format)
        df_filtrado['costo_matricula'] = df_filtrado['costo_matricula'].round(0).map('{:,.0f}'.format)
        
        print(df_filtrado.to_markdown(index=False))

    elif num_opcion == 4:
        print('Aprobados / reprobados - Programacion')
        aprobados = df_bbdd.query('nota_programacion >= 3').shape[0]
        reprobados = df_bbdd.query('nota_programacion < 3').shape[0]
        df_programacion = pd.DataFrame({
            'Aprobados': [aprobados],
            'Reprobados': [reprobados]
        })
        print(df_programacion.to_markdown(index=False))
        print('')

        print('Aprobados / reprobados - Estadística')
        aprobados = df_bbdd.query('nota_estadistica >= 3').shape[0]
        reprobados = df_bbdd.query('nota_estadistica < 3').shape[0]
        df_estadistica = pd.DataFrame({
            'Aprobados': [aprobados],
            'Reprobados': [reprobados]
        })
        print(df_estadistica.to_markdown(index=False))
        print('')


        print('Aprobados / reprobados - Matematicas')
        aprobados = df_bbdd.query('nota_matematicas >= 3').shape[0]
        reprobados = df_bbdd.query('nota_matematicas < 3').shape[0]
        df_matematicas = pd.DataFrame({
            'Aprobados': [aprobados],
            'Reprobados': [reprobados]
        })
        print(df_matematicas.to_markdown(index=False))

    elif num_opcion == 5:
        df_filtrado = df_bbdd
        costo_matricula = 200_000
        condiciones = [
            df_filtrado['cantidad_hermanos'] >= 3,
            df_filtrado['cantidad_hermanos'] == 2,
            df_filtrado['cantidad_hermanos'] == 1
        ]
        valores = [0.20, 0.15, 0.10]

        df_filtrado['descuento_pct'] = np.select(condiciones, valores, default=0.0)
        df_filtrado['costo_neto'] = costo_matricula * (1 - df_filtrado['descuento_pct'])
        total = df_filtrado['costo_neto'].sum()
        print(f'Total a pagar por todas las matrículas: {total:,.0f}')

    elif num_opcion == 6:
        df_filtrado = df_bbdd
        df_filtrado['promedio_notas'] = df_filtrado[['nota_programacion','nota_estadistica','nota_matematicas']].mean(axis=1).round(2)
        promedio_general = df_filtrado['promedio_notas'].mean(axis=0).round(2)
        df_promedio_general = pd.DataFrame({'promedio_general': [promedio_general]})

        print(df_promedio_general.to_markdown(index=False))

    elif num_opcion == 7:

        df_filtrado = df_bbdd
        df_filtrado['promedio_notas'] = df_filtrado[['nota_programacion', 'nota_estadistica', 'nota_matematicas']].mean(axis=1).round(2)
        aprobados = df_filtrado['promedio_notas'].apply(lambda x: 1 if x >= 3.0 else 0).sum()
        reprobados = df_filtrado['promedio_notas'].apply(lambda x: 1 if x < 3.0 else 0).sum()
        rendimiento = aprobados / (aprobados + reprobados)

        if rendimiento < 0.3:
            result = 'Bajo'
        elif 0.30 <= rendimiento < 0.70:
            result = 'Mediano'
        elif rendimiento >= 0.70:
            result = 'Alto'
        else:
            result = 'No identificado'
        print(f'El rendimiento academico total es {result}, con un porcentaje de: {rendimiento:.2%}')

    elif num_opcion == 8:
        print('Dinero ahorrado en descuentos (total)')

        df_filtrado = df_bbdd.copy()
        costo_matricula = 200_000

        condiciones = [
            df_filtrado['cantidad_hermanos'] >= 3,
            df_filtrado['cantidad_hermanos'] == 2,
            df_filtrado['cantidad_hermanos'] == 1
        ]
        valores = [0.20, 0.15, 0.10]

        df_filtrado['descuento_pct'] = np.select(condiciones, valores, default=0.0)

        df_filtrado['ahorro'] = (costo_matricula * df_filtrado['descuento_pct'])
        total_ahorrado = df_filtrado['ahorro'].sum()

        print(f'Total ahorrado por descuentos: {total_ahorrado:,.0f}')

    else:
        print('[ERROR] Se selecciona una opcion inexistente, por favor validar.')
    

    if input('¿Deseas realizar otra validación? (S/N): ').upper() == 'N':
        print('[INFO] Finalizando programa, ten lindo día.')
        break