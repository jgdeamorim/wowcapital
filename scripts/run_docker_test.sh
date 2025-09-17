#!/bin/bash

echo "🐳 WOW Capital - Docker Test Runner"
echo "==================================="

# Verificar se Docker está disponível
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Instale o Docker primeiro."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose não encontrado. Instale o Docker Compose primeiro."
    exit 1
fi

# Função para cleanup
cleanup() {
    echo "🧹 Limpando containers..."
    docker-compose -f docker-compose.test.yml down --remove-orphans 2>/dev/null
}

# Cleanup ao sair
trap cleanup EXIT

# Menu de opções
echo ""
echo "Selecione o tipo de teste:"
echo "1) Teste Completo (com todas as dependências)"
echo "2) Teste Simples (sem dependências externas)"
echo "3) Shell Interativo (para debug)"
echo "4) Build apenas (sem executar)"

read -p "Opção [1-4]: " option

case $option in
    1)
        echo "🚀 Executando teste completo..."
        docker-compose -f docker-compose.test.yml up --build test-runner
        ;;
    2)
        echo "🧪 Executando teste simples..."
        docker-compose -f docker-compose.test.yml up --build simple-test
        ;;
    3)
        echo "💻 Abrindo shell interativo..."
        docker-compose -f docker-compose.test.yml up --build -d dev-shell
        echo "Shell disponível em: docker exec -it wow-capital-dev bash"
        echo "Para sair: docker-compose -f docker-compose.test.yml down"
        ;;
    4)
        echo "🔨 Fazendo build apenas..."
        docker-compose -f docker-compose.test.yml build
        echo "✅ Build concluído."
        ;;
    *)
        echo "❌ Opção inválida. Use 1, 2, 3 ou 4."
        exit 1
        ;;
esac

echo ""
echo "🏁 Execução finalizada."