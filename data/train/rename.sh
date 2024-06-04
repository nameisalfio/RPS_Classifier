#!/bin/bash

# Assegna i parametri passati allo script
subdir=$1
index=$2

# Controlla se i parametri sono stati forniti
if [[ -z "$subdir" || -z "$index" ]]; then
    echo "Uso: $0 <sottocartella> <indice_iniziale>"
    exit 1
fi

# Naviga nella sottocartella
cd "$subdir" || { echo "Errore: impossibile accedere alla sottocartella"; exit 1; }

# Creiamo un array con tutti i nomi dei file ordinati per indice numerico
files=( $(ls frame_*.png | sort -V) )

# Loop attraverso tutti i file
for file in "${files[@]}"; do
    # Generiamo il nuovo nome del file con l'indice corrente
    new_name=$(printf "frame_%d.png" "$index")
    
    # Rinominiamo il file
    mv "$file" "$new_name"
    
    # Incrementiamo l'indice
    ((index++))
done

echo "I file nella sottocartella sono stati rinominati con successo."
