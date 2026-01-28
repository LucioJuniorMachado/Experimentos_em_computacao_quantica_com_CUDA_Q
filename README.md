# Experimentos_em_computacao_quantica_com_CUDA_Q

#### Aluno: Lúcio Adalberto Lima Machado Júnior (https://github.com/LucioJuniorMachado)
#### Orientadora: Leonardo Alfredo Forero Mendoza (https://github.com/leofome8)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

(https://github.com/LucioJuniorMachado/Experimentos_em_computacao_quantica_com_CUDA_Q)

---


### Resumo

Estes tutoriais fornecem exemplos práticos de algoritmos quânticos aplicados a importantes desafios na área de finanças. Nos possibilitando explorar o uso de 'quantum walks' para modelar dados financeiros, assim comparando-os aos métodos clássicos. Os exercícios também oferecem um mergulho profundo na otimização de portfólios de investimento, demonstrando soluções construídas com 'Quantum Approximate Optimization Algorithm' (QAOA), 'quantum annealing' e o inovador algoritmo QChop da Infleqtion.


### Abstract 

These tutorials provide practical examples of quantum algorithms applied to important challenges in the field of finance. They allow us to explore the use of 'quantum walks' to model financial data, comparing them to classical methods. The exercises also offer a deep dive into investment portfolio optimization, demonstrating solutions built with Quantum Approximate Optimization Algorithm (QAOA), quantum annealing, and and the novel QChop algorithm from Infleqtion.


### 1. Introdução

Esta série de tutoriais explora a interseção entre computação quântica e finanças por meio de 'notebooks' práticos usando CUDA-Q. Instituições financeiras enfrentam desafios computacionais complexos em previsão e otimização, e os recentes avanços na computação quântica estão sendo explorados pela academia e pela indústria como soluções potenciais para esses problemas tradicionalmente difíceis (Herman et al). Implementando algoritmos quânticos com CUDA-Q, obtivemos insights sobre os desafios da modelagem financeira e exploramos algoritmos híbridos quântico-clássicos, no qual podem ser aplicados a uma ampla gama de outros campos.

Nos 'Notebooks' 1 e 2, mergulhamos nos fundamentos dos 'quantum walks', vendo como eles diferem dos passeios aleatórios clássicos e aplicamos para modelar dados financeiros. O tutorial primeiro serve como uma introdução aos 'quantum walks', usando o 'quantum walk' em tempo discreto como exemplo. Ele prepara para o próximo tutorial, onde se aplica 'quantum walks' a um problema financeiro. Aprenderemos sobre 'quantum walks' comparando-os com passeios aleatórios clássicos. Ao final deste tutorial, teremos implementado um 'quantum walk' variacional em tempo discreto usando CUDA-Q, preparando o terreno para explorar protocolos de passeio mais avançados e otimização de parâmetros no próximo 'notebook'.

O 'Notebook' 3 demonstra como a computação quântica pode otimizar carteiras de investimento, explorando três abordagens diferentes: 'Quantum Approximate Optimization Algorithm' (QAOA), 'quantum annealing' e um novo algoritmo da Infleqtion chamado QChop. O objetivo da otimização de portfólio é selecionar a melhor combinação de ativos para maximizar os retornos e minimizar os riscos.


### 2. Modelagem

No início do tutorial 'Quantum Walks for Finace Part 1' criamos uma operação de porta quântica para a matriz T e chamamos-a de Tgate. Em seguida, construímos uma porta X seguida de uma porta T ao primeiro qubit. No segundo qubit conbstruímos uma porta Hadamard. Com a uma porta T fizemos um qubit de controle 0 e um qubit alvo 1. Visualizamos o núcleo quântico usando notação de circuito. No exercício 1 criamos um kernel quântico que inicializou o estado do 'quantum walker'. No exercício 2 fiz um kernel que criou um estado 'DECREMENTER' de 4 qubits |0001>. No exercício 3 alterarei o estado 'quantum walker' com base em lançamentos de moeda. 

No 'notebook Quantum Walks for Finance Part 2', definimos os kernels para impedir transições entre |0000> e |1111>, como feito no notebook 'Part 1'. O tutorial explora 'quantum qualks' múltiplos, detalhada por Chang et al., para carregar uma distribuição de probabilidade específica em um estado quântico. O foco aqui é a distribuição log-normal, que pode modelar o preço à vista de um ativo financeiro no vencimento. Preparando os dados, comparamos 'quantum walks' com a distribuição alvo. A seguir trabalhamos 'quantum walks' em passos divididos, quando vira à esquerda e quando à direita. No exercício 1 codificamos uma etapa do SSQW usando duas operações de moedas diferentes F1=X e F2=H e não permitindo o movimento do 'walker' entre a posição |0000⟩ e |1111⟩. Então. executamos o multi-SSQW e plotamos os resultados, como o resultado da medição do estado de Bell. Como desafio  reescrevemos o penúltimo código para que agora funcione com um número arbitrário de qubits usando a variável 'num_qubits', e adaptamos o código e criamos um 'Quantum Walk-Based Adaptive Distribution Generator' que permitiu modelar não apenas dados financeiros, mas também dados 2D, como imagens pixelizadas de dígitos manuscritos.

No terceiro tutorial, o 'notebook' Portifólio de Otimização, iremos  configurar um problema de otimização, resolvendo-o usando dois métodos quânticos diferentes, adicionando restrições e explorando métodos para escalar essas abordagens. Este tutorial também explorará o algoritmo de última geração Q-CHOP, desenvolvido pela Infleqtion e por JPMorgan Chase, demonstrando que exemplos fundamentais codificados com CUDA-Q podem ser estendidos para aplicações úteis dia-a-dia. Já no primeiro exercício, dados os valores de α, β, μ e σ escrevemos o código gerando a matriz QUBO Q. Após, Agora, adicionamos um código que gerou todos os portfólios possíveis e avaliamos a qualidade do portfólio calculando manualmente o xTQx e retornando o melhor e o pior portfólios. No segundo exercício, construímos um Hamiltoniano de Ising e produzimos um Hamiltoniano de operador de spin CUDA-Q. A primeira função recebeu uma matriz QUBO triangular superior como um array numpy e retornou uma lista de coeficientes para os termos de uma única variável. Também, escrevemos uma segunda função que utilizava essas entradas e imprimimos o Hamiltoniano para confirmar se estava correto. No exercício terceiro, codificamos um kernel QAOA em CUDA-Q. Observamos como as entradas são fornecidas como listas, de acordo com suas funções anteriores. Os elementos da função de custo são aplicados como portas RZ parametrizadas com um γ para cada camada, multiplicado por hi para os termos de variável única. Os termos de duas variáveis ​​são aplicados como uma porta CNOT entre os dois qubits correspondentes (controlados pelo primeiro), uma porta RZ parametrizada com o mesmo γ para cada camada, multiplicado por Jij, seguida por uma repetição da porta CNOT anterior. Os termos do misturados são portas RX simples aplicadas a cada qubit, parametrizadas com 2,0∗β para cada camada. Executamos um script com resultados de amostras e confirmamos se o portifólio ideal está pelo menos entre os 5 melhores resultados amostrados pelo procedimento QAOA. Para visualizar melhor o desempenho da sua função QAOA, usamos a função 'plot_samples_histogram' para plotar uma amostra do kernel QAOA com seus parâmetros iniciais (uma superposição uniforme, já que todos os parâmetros variacionais são 0) e o estado final obtido. Rotulamos como "Bons Portfólios" os resultados que correspondem a um valor HC negativo. No quarto exercício, a aproximação adiabática, confirmamos se a abordagem funciona. Observamos como a variável T controla a frequência com que a evolução ocorre. Tentamos executar novamente com T=10. O resultado foi uma evolução moderada, que é: melhoria significativa em relação a T=1; portfólios de bom desempenho amplificados; e melhor índice de aproximação. No quinto exercício adicionamos restrições, escrevemos uma nova função 'portfolio_to_qubo', que adiciona restrições para limitar as soluções de portfólio a k ações. Usando todas as funções criadas anteriormente, executamos novamente uma simulação adiabática desse resultado com restrições. O histograma produzido indicou que a restrição funcionou para números de ações que sejam 2. No sexto exercício, melhorando a Convergência Adiabática com Q-CHOP, usamos um código que gerou μ e σ aleatórios para n=15 ações. Usamos suas funções anteriores para executar um QAOA. Executamos quatro amostras diferentes do estado final com 50, 100, 500 e 1000 iterações e verificamos quantas vezes o portifólio ótimó é amostrado. Para este tamanho de problema relativamente modesto, não existe o risco dela ser perdida com 1000 'shots' (<1% de chance). 


### 3. Resultados

No tutorial 1, criamos um kernel que inicializou o 'quantum walk'. 

<img width="786" height="535" alt="Captura de tela 2026-01-28 115822" src="https://github.com/user-attachments/assets/5efef878-259a-4b54-aabc-ee46a3f07e2d" />

Definimos uma operação personalizada em 4 qubits para a matriz unitária INC que
mapeia |x> para |x+1> mod 16 e verificamos se ela funciona como esperado para |0000>.

<img width="264" height="182" alt="Captura de tela 2026-01-28 120105" src="https://github.com/user-attachments/assets/6489bdd7-a012-48ad-aa78-de7796d3b078" />

No tutorial 2, preparamos os dados e comparamos 'quantum walk' e a distribuição alvo.

<img width="931" height="222" alt="Captura de tela 2026-01-28 120627" src="https://github.com/user-attachments/assets/130411d9-9577-4134-ad44-a22734462bb6" />

No exercício 1, codificamos uma etapa do SSQW abaixo usando duas operações de moeda diferentes,  impedindo o movimento do caminhante entre as posições |0000> e |1111>, caso a moeda seja de |0> se lançando para |1>. 

<img width="645" height="470" alt="Captura de tela 2026-01-28 121021" src="https://github.com/user-attachments/assets/7cdb07c6-0771-438f-a2cf-4f7275059573" />

Testamos o 'quantum walk' kernel e mesuramos o estado de Bell. 

<img width="879" height="486" alt="Captura de tela 2026-01-28 121135" src="https://github.com/user-attachments/assets/7778ca7d-e452-44d5-82f3-33244a0dc681" />

Em cenários reais criamos 'quantum walks' e demostramos uma simples 'quantum walk'. 

<img width="950" height="285" alt="Captura de tela 2026-01-28 121454" src="https://github.com/user-attachments/assets/7b72456f-fa35-4ce5-8445-cc2148708a15" />

<img width="961" height="362" alt="Captura de tela 2026-01-28 121528" src="https://github.com/user-attachments/assets/bf2378d8-e852-40aa-aad2-77752bad1420" />

Comparamos 'quantum walks' com as distribuições alvo e seus erros de distribuições. Listamos os top cinco 'quantum walks'.

<img width="981" height="660" alt="Captura de tela 2026-01-28 122007" src="https://github.com/user-attachments/assets/b30b0cac-83ff-43f4-9f6a-287b15bbe723" />

<img width="1020" height="652" alt="Captura de tela 2026-01-28 122326" src="https://github.com/user-attachments/assets/8b96073c-23a9-418e-8de5-9fb01267564f" />

Modelamos dados em 2D, imagens pixelizadas de dígitos manuscritos.

<img width="1020" height="852" alt="Captura de tela 2026-01-28 122623" src="https://github.com/user-attachments/assets/9850d784-6794-42f8-af09-28b0659eb2d8" />

Aqui o conjunto de distribuição dos 'quantum walks'

<img width="971" height="531" alt="Captura de tela 2026-01-28 122752" src="https://github.com/user-attachments/assets/0fab92c0-e6e7-44a7-b1c5-0d190c594a2e" />

Através QUBO matrix Q no terceiro tutorial apresentamos os melhores portifólios e o pior. 

<img width="420" height="710" alt="Captura de tela 2026-01-28 124150" src="https://github.com/user-attachments/assets/6ab530ab-a2ac-4c8a-bea9-4fc54993488a" />

No exercício 2 escrevemos uma segunda função com as entradas e construímos o Hamiltoniano como um objeto operador de spin CUDA-Q. Imprimimos o Hamiltoniano para confirmar se estava correto.

<img width="935" height="303" alt="Captura de tela 2026-01-28 124450" src="https://github.com/user-attachments/assets/cf7711ce-d216-4375-a218-cc0d1e86b10b" />

No exercício 3 salvamos os resultados de amostragem do circuito e mostramos os melhores portifólios. 

<img width="688" height="766" alt="Captura de tela 2026-01-28 124752" src="https://github.com/user-attachments/assets/6ec7fa8c-259b-4463-8eb1-dea25effc96a" />

<img width="886" height="542" alt="Captura de tela 2026-01-28 124838" src="https://github.com/user-attachments/assets/01a22e99-51ce-4ff0-8600-5e8955af26f5" />

Executamos o script abaixo para ver os resultados de amostra e confirmar se o portifólio ideal está pelo menos entre os 5 melhores resultados amostrados pelo procedimento QAOA.

<img width="957" height="666" alt="Captura de tela 2026-01-28 125157" src="https://github.com/user-attachments/assets/97f23970-b358-4177-b460-759effd2678e" />

Na otimização de portifólios com QAOA obtivemos os resultados: 

<img width="924" height="844" alt="Captura de tela 2026-01-28 125425" src="https://github.com/user-attachments/assets/7f886211-46a4-4f93-868a-0b2efad720fb" />

<img width="941" height="646" alt="Captura de tela 2026-01-28 125453" src="https://github.com/user-attachments/assets/5c8bd043-f4bb-4908-8bf7-a3bb527ff8d4" />

No exercício 4 vimos a simulação da evolução adiabática e que todos os portifólios são iqualmente prováveis. 

<img width="496" height="750" alt="Captura de tela 2026-01-28 125715" src="https://github.com/user-attachments/assets/4859d2d9-86e7-4535-b773-bb9f235b06a7" />

<img width="984" height="820" alt="Captura de tela 2026-01-28 125805" src="https://github.com/user-attachments/assets/62bc677f-b72b-40ab-958a-5ffb4bdef773" />

No exercício de otimização de portfólio restrito, os mais prováveis portifólios foram com duas ações. Aqui os resultados: 

<img width="976" height="733" alt="Captura de tela 2026-01-28 130047" src="https://github.com/user-attachments/assets/78ca0ccd-e8e2-45a8-8fdb-6915a278886a" />

<img width="970" height="662" alt="Captura de tela 2026-01-28 130157" src="https://github.com/user-attachments/assets/e57b996d-30c4-490e-a8ae-9cd1decbe272" />

No exercício 6, através do cíodigo geramos valores aleatórios para ações. Executamos quatro amostras diferentes do estado final com 50, 100, 500 e 1000 iterações e verificamos quantas vezes a portifólio ótimo foi amostrado. 

<img width="601" height="469" alt="Captura de tela 2026-01-28 130623" src="https://github.com/user-attachments/assets/f2df0e27-736f-4da1-af31-046ea69b340d" />

<img width="620" height="748" alt="Captura de tela 2026-01-28 130654" src="https://github.com/user-attachments/assets/3736ab35-6f57-49b5-969a-b8f2c851022e" />

<img width="773" height="223" alt="Captura de tela 2026-01-28 130733" src="https://github.com/user-attachments/assets/0ade2e06-0376-48a0-bc09-0672983b2f6d" />

<img width="985" height="699" alt="Captura de tela 2026-01-28 130803" src="https://github.com/user-attachments/assets/8f37871e-4c2e-463a-9818-e8cc3bb74909" />

<img width="940" height="650" alt="Captura de tela 2026-01-28 130942" src="https://github.com/user-attachments/assets/23738411-753e-4d9a-82ae-ad1ac735efeb" />

Executamos a otimização QAOA para n=15 ações. 

<img width="660" height="414" alt="Captura de tela 2026-01-28 131148" src="https://github.com/user-attachments/assets/b3a1dc36-f810-4246-83e5-429a9ae8c6a7" />

<img width="957" height="672" alt="Captura de tela 2026-01-28 131215" src="https://github.com/user-attachments/assets/0e2a098e-b1e0-4cb0-b436-605f8855b34d" />

Produzimos amostras e contabilizamos a ocorrência de portifólios ideais. 

<img width="625" height="801" alt="Captura de tela 2026-01-28 131414" src="https://github.com/user-attachments/assets/bc64b47d-a2c8-48a5-9a12-1f063100e4af" />

<img width="961" height="698" alt="Captura de tela 2026-01-28 131442" src="https://github.com/user-attachments/assets/6e3669d1-ec09-4514-9a98-89f43c47940c" />


### 4. Conclusões

Os resultados da criação do estado de Bell no tutorial segundo foram as probabilidades: 

  |11⟩: 0.514
  |00⟩: 0.486

Os resultados do 'quantum walk'  foramr: [0.10247399, 0.13724316, 0.19850243, 0.26142815, 0.13545552, 0.0888489 e 0.07604784. Com o erro de otimização em 50 interações de 0.033003. 

O desafio teve como resultado com 2 qubits em 4 estados mostrado no quadro abaixo. E com 3 qubits com 8 estados deu erro. 

<img width="569" height="309" alt="Captura de tela 2026-01-28 140526" src="https://github.com/user-attachments/assets/867a37c3-3e9e-408b-90f6-8177a0cbc931" />

<img width="878" height="238" alt="Captura de tela 2026-01-28 140643" src="https://github.com/user-attachments/assets/618ffe66-ce52-4b19-ab5c-859bc394219e" />

Para carregar a imagem do dígito 3 do Mnist, foram precisos 6 qubits e 64 estados.

<img width="915" height="566" alt="Captura de tela 2026-01-28 140910" src="https://github.com/user-attachments/assets/bae39cf7-44fb-4f10-a00f-c2ce8ae091ed" />

No exercício 3 do terceiro tutorial, a busca exaustiva clássica encontrou o verdadeiro portfólio ótimo, os resultados simulados do QAOA mostram alta probabilidade para portfólios de baixo custo e na prática, com CUDA-Q real, o QAOA otimizou os parâmetros para preparar um estado quântico que, quando medido, fornece o portfólio ótimo com alta probabilidade.

<img width="943" height="647" alt="Captura de tela 2026-01-28 141432" src="https://github.com/user-attachments/assets/504efd39-085c-4e9e-a705-057f6574aa65" />

A seguir concluímos que mesmo que a QAOA não produza o portifólio ótimo com a maior frequência, ele amplifica significativamente a probabilidade de boas carteiras em comparação com a amostragem aleatória. O portifólio ótimo deve aparecer com muito mais frequência com a QAOA do que com a amostragem aleatória uniforme.

Os gráficos abaixo mostram estado inicial: distribuição uniforme (todas os portifólios com a mesma probabilidade); estado final: probabilidade concentrada em portifólios de baixo custo (boas); o QAOA amplifica com sucesso as probabilidades de boas soluções; e o portifólio ótimo aparece com muito mais frequência após o QAOA. Mesmo que não seja perfeito, o QAOA proporciona uma melhoria significativa em relação à amostragem aleatória.

<img width="940" height="649" alt="Captura de tela 2026-01-28 141938" src="https://github.com/user-attachments/assets/61b13647-a371-464f-b01e-b1516adbe375" />

No exercício 4, a aproximação adiabática, o teorema adiabático nos diz que, para alcançar com sucesso o estado fundamental Hamiltoniano do problema, precisamos evoluir lentamente o suficiente. Principais observações da nossa simulação: 1) T=1 (Evolução rápida): resultados próximos à distribuição uniforme; aproximação ruim da solução ótima; e condição adiabática não satisfeita. 2) T=10 (Evolução moderada): melhoria significativa em relação a T=1; portfólios bons são amplificados; e melhor taxa de aproximação. 3) T=100 (Evolução lenta): próximo ao estado fundamental ótimo; alta probabilidade de portfólios bons; e melhor taxa de aproximação. Contra-argumento: mm T maior proporciona melhores resultados, mas requer mais tempo de computação. Na prática, escolhemos T com base nos recursos disponíveis e na precisão necessária.

<img width="996" height="773" alt="Captura de tela 2026-01-28 142427" src="https://github.com/user-attachments/assets/2aa927b9-2249-4a3c-b722-c3c14527ae7f" />

Na parte que adicionamos restrições, a formulação QUBO restrita adiciona um termo de penalização que desencoraja
portfólios com qualquer número de ações que não seja exatamente 2. Observações principais: 1) A intensidade da penalização λ=10 controla a força com que a restrição é aplicada. 2) O tempo de evolução T=100 afeta a capacidade do sistema quântico de encontrar estados de baixa energia, respeitando as restrições. 3) No histograma, os portfólios com exatamente 2 ações devem ser destacados. 4) Com parâmetros adequados, a maioria dos portfólios amostrados deve ter 2 ações.

<img width="997" height="761" alt="Captura de tela 2026-01-28 142730" src="https://github.com/user-attachments/assets/4eed5a52-8f9f-4eef-aa1e-01c12f6c004a" />

Para n=15 ações com QAOA concluímos: 1) A carteira ótima tem probabilidade muito baixa sob amostragem uniforme (1 em 32.768 ≈ 0,00003052). 2) O QAOA amplifica significativamente a probabilidade de boas carteiras (tipicamente, amplificação de 100 a 1000 vezes para boas soluções). 3) Com um número moderado de tentativas:- 50 tentativas: alto risco de não encontrar a carteira ótima (>30% de chance);- 100 tentativas: risco moderado (~10-20% de chance);- 500 tentativas: baixo risco (<5% de chance);- 1000 tentativas: risco muito baixo (<1% de chance). 4) Recomendação para n=15: - use pelo menos 500 a 1000 tentativas para uma amostragem confiável; - considere múltiplas execuções independentes; - o portifólio ótimo pode não ser a mais provável, mas
deve aparecer entre os melhores resultados. 5) Para n maior (20+ ações), o risco aumenta drasticamente e são necessários números de disparos muito maiores ou algoritmos melhores.

<img width="989" height="634" alt="Captura de tela 2026-01-28 143341" src="https://github.com/user-attachments/assets/220348d6-6cac-4635-91e5-9a03590780a2" />

Os resultados da otimização QAOA para n=15 ações:

Número de camadas QAOA: 3
Total de parâmetros otimizados: 6
Iterações de otimização: 54
Valor esperado final: 0,383403
Status de sucesso: Verdadeiro

Resultados da Amostragem (1000 execuções cada):
Estado inicial - Portfólio ótimo: 128 vezes (12,80%)
Estado otimizado - Portfólio ótimo: 128 vezes (12,80%)
✗ O QAOA não melhorou a probabilidade do portfólio ótimo

Possíveis razões:

- Número insuficiente de camadas QAOA (p={camadas});

- Mínimo local na otimização;

- Necessidade de mais execuções para estatísticas confiáveis.

Principais conclusões:
1. Os parâmetros do QAOA precisam de inicialização adequada (não todos zeros);
2. A otimização requer múltiplas iterações para convergir;
3. O portfólio ótimo deve se tornar mais provável após a otimização;
4. Para n=15, mesmo A otimização do QAOA pode não tornar o ideal o mais provável;
5. Mas deve aparecer com muito mais frequência do que no estado inicial.

<img width="965" height="602" alt="Captura de tela 2026-01-28 143656" src="https://github.com/user-attachments/assets/757d354f-fb14-4125-b380-36391dbe470d" />

RESUMO E CONCLUSÃO

Para n=15 ações (2^15 = 32.768 carteiras possíveis):

Análise de Contagem de 'shots':
'Shots' Probabilidade Ótima Amplificação Risco de Erro

50 22 0,440000 14417,9x 0,000
100 30 0,300000 9830,4x 0,000
500 78 0,156000 5111,8x 0,000
1000 128 0,128000 4194,3x 0,000

Principais conclusões:
1. O QAOA amplifica significativamente a probabilidade de encontrar boas carteiras
(tipicamente de 5112 a 4194 vezes melhor do que um palpite aleatório).

2. O risco de não encontrar a carteira ideal diminui com o número de tentativas:

- 50 tentativas: Alto risco (probabilidade de erro de aproximadamente 30 a 50%);

- 100 tentativas: Risco moderado (probabilidade de erro de aproximadamente 10 a 20%);

- 500 tentativas: Baixo risco (probabilidade inferior a 5%);

- 1000 tentativas: Risco muito baixo (probabilidade inferior a 1%).

3. Mesmo que a carteira ideal não seja A mais provável, ela aparece com frequência suficiente
para ser identificada com amostragem adequada.

4. Para aplicações práticas com n=15:
- 500 a 1000 simulações fornecem resultados confiáveis
- Múltiplas execuções podem reduzir ainda mais o risco
- Considere o equilíbrio entre o número de simulações e o número de execuções de otimização

RECOMENDAÇÃO FINAL

Para otimização de portfólio com n=15 ações usando QAOA:
- Use pelo menos 500 simulações para uma amostragem confiável;
- Considere 1000 ou mais simulações Ações para aplicações críticas;
- O portfólio ideal deve aparecer entre os melhores resultados;
- Mesmo que não seja o mais provável, será amostrado com frequência suficiente;
- O risco de não encontrar o ideal cai para menos de 1% com 1000 tentativas.

<img width="987" height="634" alt="Captura de tela 2026-01-28 144029" src="https://github.com/user-attachments/assets/4e96d03a-ed48-4d41-8b76-3c063d1e409c" />

CONCLUSÃO FINAL: 

A execução paralela com múltiplas QPUs proporciona:
1. Tempo de execução mais rápido através da distribuição da carga de trabalho;
2. Maior taxa de transferência para amostragem em larga escala;
3. Escalabilidade para instâncias de problemas ainda maiores.

Para 8.000.000 de 'shots' (disparos):

- Serial: 8,00 segundos;
- Paralelo (4 QPUs): 2,50 segundos;
- Aceleração: 3,20x mais rápido.

Considerações práticas:
- Use execução paralela para grandes quantidades de disparos (>1 milhão);
- Distribua a carga de trabalho uniformemente entre as QPUs;
- Combine os resultados após a execução para análise;
- Monitore a utilização das QPUs para um desempenho ideal.


---

Matrícula: 232.100.431

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*

