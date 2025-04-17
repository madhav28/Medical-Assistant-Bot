#!/bin/bash --login
# Download PubMed data for 2023-06

output_file="/mnt/scratch/lellaom/corewell/pubmed_2023_06.txt"

# Set the API key as an environment variable
export NCBI_API_KEY="7dfcbe4ed2ad28246c86e4ca0c947d23cb09"

echo "Running esearch..."
esearch -db pubmed -query "2023/06/01[PDAT] : 2023/06/30[PDAT]" | \
efetch -format abstract > "$output_file"

# Add a delay to avoid overloading the server
sleep 1

if [ -f "$output_file" ]; then
    echo "Publications from 2023-06 have been downloaded to $output_file."
else
    echo "Failed to download publications for 2023-06. Please check your EDirect setup and try again."
fi
