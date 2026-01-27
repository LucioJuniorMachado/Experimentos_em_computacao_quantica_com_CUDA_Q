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

Esta série de tutoriais explora a interseção entre computação quântica e finanças por meio de notebooks práticos usando CUDA-Q. Instituições financeiras enfrentam desafios computacionais complexos em previsão e otimização, e os recentes avanços na computação quântica estão sendo explorados pela academia e pela indústria como soluções potenciais para esses problemas tradicionalmente difíceis (Herman et al). Implementando algoritmos quânticos com CUDA-Q, obtivemos insights sobre os desafios da modelagem financeira e exploramos algoritmos híbridos quântico-clássicos, no qual podem ser aplicados a uma ampla gama de outros campos.

Nos 'Notebooks' 1 e 2, mergulhamos nos fundamentos dos passeios quânticos, vendo como eles diferem dos passeios aleatórios clássicos e aplicamos para modelar dados financeiros. O tutorial primeiro serve como uma introdução aos 'quantum walks', usando o 'quantum walk' em tempo discreto como exemplo. Ele prepara para o próximo tutorial, onde se aplica 'quantum walks' a um problema financeiro. Aprenderemos sobre 'quantum walks' comparando-os com passeios aleatórios clássicos. Ao final deste tutorial, teremos implementado um 'quantum walk' variacional em tempo discreto usando CUDA-Q, preparando o terreno para explorar protocolos de passeio mais avançados e otimização de parâmetros no próximo 'notebook'.

O 'Notebook' 3 demonstra como a computação quântica pode otimizar carteiras de investimento, explorando três abordagens diferentes: 'Quantum Approximate Optimization Algorithm' (QAOA), 'quantum annealing' e um novo algoritmo da Infleqtion chamado QChop. O objetivo da otimização de portfólio é selecionar a melhor combinação de ativos para maximizar os retornos e minimizar os riscos.


### 2. Modelagem

No início do notebook 'Quantum Walks for Finace Part 1' criei uma operação de porta quântica para a matriz T e chamei-a de Tgate. Em seguida, construi uma porta X seguida de uma porta T ao primeiro qubit. No segundo qubit conbstrui uma porta Hadamard. Com a uma porta T fiz um qubit de controle 0 e um qubit alvo 1. Visualizei o núcleo quântico usando notação de circuito. No exercício 1 criei um kernel quântico que inicializou o estado do 'quantum walker'. No exercício 2 fiz um kernel que criou um estado 'DECREMENTER' de 4 qubits |0001>. No exercício 3 alterarei o estado 'quantum walker' com base em lançamentos de moeda. 

No notebook 'Quantum Walks for Finance Part 2', defini os kernels para impedir transições entre |0000> e |1111>, como feito no notebook 'Part 1'. O tutorial explora 'quantum qualks' múltiplos, detalhada por Chang et al., para carregar uma distribuição de probabilidade específica em um estado quântico. O foco aqui é a distribuição log-normal, que pode modelar o preço à vista de um ativo financeiro no vencimento. Preparando os dados, comparamos 'quantum walks' com a distribuição alvo. A seguir trabalhamos 'quantum walks' em passos divididos, quando vira à esquerda e quando à direita. No exercício 1 codificamos uma etapa do SSQW usando duas operações de moedas diferentes F1=X e F2=H e não permitindo o movimento do 'walker' entre a posição |0000⟩ e |1111⟩. Então. executei o multi-SSQW e plotei os resultados, como o resultado da medição do estado de Bell. Como desafio  reescrevi o peníultimo código para que agora funcione com um número arbitrário de qubits usando a variável 'num_qubits', e adaptei o código e criar um 'Quantum Walk-Based Adaptive Distribution Generator' que permitiu modelar não apenas dados financeiros, mas também dados 2D, como imagens pixelizadas de dígitos manuscritos.




### 3. Resultados

No início do notebook 'Quantum Walks for Finace Part 1' criei uma operação de porta quântica para a matriz T e chamei-a de Tgate. Em seguida, construi uma porta X seguida de uma porta T ao primeiro qubit. No segundo qubit conbstrui uma porta Hadamard. Com a uma porta T fiz um qubit de controle 0 e um qubit alvo 1. Visualizei o núcleo quântico usando notação de circuito. Após, criei um kernel quântico que inicializou o estado do 'quantum walker'.

### 4. Conclusões

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin pulvinar nisl vestibulum tortor fringilla, eget imperdiet neque condimentum. Proin vitae augue in nulla vehicula porttitor sit amet quis sapien. Nam rutrum mollis ligula, et semper justo maximus accumsan. Integer scelerisque egestas arcu, ac laoreet odio aliquet at. Sed sed bibendum dolor. Vestibulum commodo sodales erat, ut placerat nulla vulputate eu. In hac habitasse platea dictumst. Cras interdum bibendum sapien a vehicula.

Proin feugiat nulla sem. Phasellus consequat tellus a ex aliquet, quis convallis turpis blandit. Quisque auctor condimentum justo vitae pulvinar. Donec in dictum purus. Vivamus vitae aliquam ligula, at suscipit ipsum. Quisque in dolor auctor tortor facilisis maximus. Donec dapibus leo sed tincidunt aliquam.

---

Matrícula: 232.100.431

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*

