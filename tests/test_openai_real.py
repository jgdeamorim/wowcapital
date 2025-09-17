#!/usr/bin/env python3
"""
WOW Capital - Teste OpenAI com Chaves Reais
Usage: OPENAI_API_KEY=your-key python test_openai_real.py
"""

import os
import sys
from openai import OpenAI

def test_openai_real():
    print('🔑 WOW CAPITAL - TESTE OPENAI COM CHAVES REAIS')
    print('=' * 55)

    # Verificar API Key
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print('❌ OPENAI_API_KEY não configurada!')
        print('\n📝 Para testar, execute:')
        print('   export OPENAI_API_KEY="sua-chave-aqui"')
        print('   python test_openai_real.py')
        return False

    print(f'✅ API Key encontrada: {api_key[:20]}...')

    try:
        # Inicializar cliente
        client = OpenAI(api_key=api_key)
        print('✅ Cliente OpenAI inicializado')

        # Teste 1: Chamada simples
        print('\n🧪 TESTE 1 - Chamada Simples:')
        print('-' * 55)

        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'system',
                    'content': 'Você é um assistente de trading quantitativo para WOW Capital.'
                },
                {
                    'role': 'user',
                    'content': 'Confirme que está funcionando dizendo: "WOW Capital OpenAI 100% operacional!"'
                }
            ],
            max_tokens=50,
            temperature=0
        )

        print('✅ Resposta:')
        print(f'   {response.choices[0].message.content}')
        print(f'✅ Tokens usados: {response.usage.total_tokens}')

        # Teste 2: Análise de mercado simulada
        print('\n🧪 TESTE 2 - Análise Técnica:')
        print('-' * 55)

        response2 = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'system',
                    'content': 'Você é um analista quantitativo especializado.'
                },
                {
                    'role': 'user',
                    'content': 'Analise brevemente: BTC subiu 5% em 1h com volume 2x acima da média. Uma frase.'
                }
            ],
            max_tokens=100,
            temperature=0.3
        )

        print('✅ Análise:')
        print(f'   {response2.choices[0].message.content}')
        print(f'✅ Tokens usados: {response2.usage.total_tokens}')

        # Teste 3: Validação de embeddings
        print('\n🧪 TESTE 3 - Embeddings:')
        print('-' * 55)

        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="WOW Capital trading strategy optimization"
        )

        embedding_vector = embedding_response.data[0].embedding
        print(f'✅ Embedding gerado: {len(embedding_vector)} dimensões')
        print(f'✅ Primeiros 5 valores: {embedding_vector[:5]}')

        print('\n🎉 TODOS OS TESTES PASSARAM!')
        print('✅ OpenAI: 100% Funcional com chaves reais')
        print('✅ Chat Completions: OK')
        print('✅ Embeddings: OK')
        print('✅ Pronto para integração com WOW Capital')

        return True

    except Exception as e:
        error_msg = str(e)
        print(f'❌ Erro: {error_msg}')

        if 'api_key' in error_msg.lower():
            print('   → Verifique se a API key está correta')
        elif 'rate_limit' in error_msg.lower():
            print('   → Aguarde alguns minutos (rate limit)')
        elif 'quota' in error_msg.lower():
            print('   → Verifique seu billing na OpenAI')
        elif 'insufficient_quota' in error_msg.lower():
            print('   → Adicione créditos à sua conta OpenAI')
        else:
            print('   → Erro de conectividade')

        return False

if __name__ == '__main__':
    success = test_openai_real()
    sys.exit(0 if success else 1)