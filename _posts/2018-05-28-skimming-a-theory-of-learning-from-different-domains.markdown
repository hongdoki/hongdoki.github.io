---
layout: post
title: 도메인 적응 이론 논문 훑어보기 - "A theory of learning from different domains"
comments: true
---

어떤 공장에서는 오랜 기간 **제품 A**를 생산해왔고, 생산 과정에서 발생한 결함 **데이터를 기반**으로 **결함을 검출하는 분류기**를 학습하여 사용하고 있었다. 그런데 시대가 변해 제품 A와는 **비슷하지만 조금 다른 제품 B**를 생산해야 하는 상황이 발생하였는데, 제품 B는 신제품이다 보니 분류기를 학습하기 위한 데이터가 적어 분류기의 성능이 낮다는 문제가 발생하였다. 만약 분류기가 아니라 사람이라면 제품 A에서 검사하던 경험을 살려서 제품 B와 관련한 적은 경험을 가지고도 높은 정확도의 결함 검사가 가능할 것이다. 우리의 분류기도 이처럼 사용할 수 없을까?

## 도메인 적응(Domain Adaptation)

도메인 적응은 머신러닝 문제에서 위의 예처럼 적용되던 **영역(Domain)**이 약간 달라졌을 때, **다르지만 관련 있는** 새로운 영역에 기존 영역의 정보를 **적응(Adaptation)**시켜서 사용하고자 하는 목적을 가지고 있는 연구 분야이다. 영역이라는 것은 해석하기 조금 모호한 용어인데, 구체적으로 데이터의 분포를 생각하면 이해하기 쉽다. 기존에 모델이 동작하던 영역을 **소스(source) 도메인**이라고 하고, 새로운 영역을 **타겟(target) 도메인**이라고 한다. 특정 작업이나 영역이 바뀌었을 때 기존의 정보를 잘 전이(Transfer)하여 활용하는 것을 **전이 학습(Transfer Learning)**라고 하는데 도메인 적응은 전이 학습의 하위 분야이다. 

## "A theory of learning from diffrent domains"

도메인 적응을 하기 위한 실험적인 방법론들이 많이 연구되고 있다. 수많은 논문들을 접하면서 근본적인 이론을 알고 있다면 획기적인 방법을 찾을 수 있지 않을까 하여 읽었던 논문이 2010년에 발표된 **S. Ben-David**의 **"A theory of learning from different domains"**이라는 논문이다. 결론적으로 해당 논문을 통해 실용적인 방법론을 얻지는 못했지만 머신러닝을 공부할 때 PAC learning에 해당하는 내용을 공부하듯이 해당 논문의 내용을 알아둔다면 도메인 적응에 관한 근본적이고 질문에 개념적으로 설명을 하는 데 도움이 될 수 있을 것 같아 정리하고자 해당 논문을 훑어보는 글을 쓰게 되었다.

해당 논문은 도메인 적응을 하게 되면 근본적이라고 생각되는 두 가지 질문에 대한 답을 얻고자 한다. 질문은 아래와 같다.

1. 소스 도메인에서 학습한 분류기는 어떤 조건일 때 타겟 도메인에서도 잘 작동할 것인가?
2. 소스 도메인의 데이터(이하 소스 데이터)와 더불어 타겟 도메인의 데이터(이하 타겟 데이터)가 있을 때 어떻게 이를 이용하면 타겟 도메인에서 분류기의 오류(error)를 낮출 수 있을 것인가?

### 문제 및 표기법 정의

첫 번째 질문에 대해 대답하기 전에 우리가 풀고자 하는 문제와 표기법에 대해 정의를 하자. 문제의 단순화를 위해 클래스가 두 개인 **이진 분류 문제**를 가정하면, 입력값 $$\mathcal{X}$$를 받아서 항상 정답만을 출력하는 가상의 함수인 **레이블링 함수**를 $$f:\mathcal{X}\to\{0,1\}$$로 표현할 수 있다. 우리의 목표는 레이블링 함수를 를 완벽히 모사하는 함수인 **가설(hypothesis)** $$h:\mathcal{X}\to\{0,1\}$$을 만드는 것이다. 그리고 도메인은 **분포(distribution)** $$\mathcal{D}$$와 레이블링 함수로 이루어져 있는데, **소스 도메인**은 $$\langle\mathcal{D}_S, f_S\rangle$$로 **타겟 도메인**은$$\langle\mathcal{D}_T, f_T\rangle$$ 로 표기할 수 있다. 이때 우리가 만든 가설 $$h$$가 $$\mathcal{D}_S$$의 분포를 따르는 입력값을 받아서 출력한 결과값과 정답과의 차이를 **소스도메인에서의 오류**(이하 **소스오류**)로 정의하는데 이는 아래와 같이 표현할 수 있다.

$$\epsilon_S(h,f) = E_{\mathbf{x}\sim \mathcal{D}_S}[|h(\mathbf{x}) - f(\mathbf{x})|]$$

위 식에서 $$S$$를 $$T$$로 바꾸면 이는 **타겟 도메인에서의 오류**(이하 **타겟 오류**)가 된다.

### 첫 번째 질문

#### 정리 1: 타겟 오류에 대한 상계

해당 논문에서는 첫 번째 질문에 대한 대답으로 정리(theorem) 1을 제시한다.

> 정리 1
>
> 가설 $$h$$에 대해,
> $$\epsilon_T(h) \le \epsilon_S(h) + d_1(\mathcal{D}_S, \mathcal{D}_T) + \min\{E_{\mathcal{D}_S}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|], E_{\mathcal{D}_T}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|]\}$$ 
>
> 여기서 $$d_1(\mathcal{D}_S, \mathcal{D}_T)$$은 두 분포 간의 거리를 나타내는 가장 자연스러운 척도인 **$$L^1$$ 다이버전스(divergence)**로, $$\mathcal{D}$$와 $$\mathcal{D'}$$하의 모든 측정 가능한 부분 집합들의 집합을 $$\mathcal{B}$$라고 할 때 아래 식과 같다.
>
> $$d_1(\mathcal{D}, \mathcal{D'})=2\sup_{B\in\mathcal{B}}|\mathrm{Pr}_\mathcal{D}[B] - \mathrm{Pr}_\mathcal{D'}[B]|$$

정리1의 부등식은 최소화하고자 하는 **타겟 오류**($$\epsilon_T(h)$$)의 최댓값을 **소스 오류**($$\epsilon_S(h)$$)와 **두 분포 간의 거리**($$d_1(\mathcal{D}_S, \mathcal{D}_T)$$), 그리고 **두 분포에서의 레이블링 함수의 차이**($$\min\{E_{\mathcal{D}_S}[\vert f_S(\mathbf{x}) - f_T(\mathbf{x})\vert], E_{\mathcal{D}_T}[\vert f_S(\mathbf{x}) - f_T(\mathbf{x})\vert]\}$$)로 제한할 수 있음을 알려주고 있다. 

즉 우리가 했던 첫번째 질문에 대한 대답으로 **어떤 분류기가 소스도메인에서 오 분류율이 낮고, 소스와 타겟 분포 간의 거리가 가깝다면 타겟 도메인에서도 잘 작동한다**고 해석할 수 있다. 단, 두 도메인의 레이블링 함수의 차이가 입력 도메인에 상관없이 적어야 한다는 것은 **두 도메인에서 모두 잘 동작하는 가상의 정답 분류기가 존재한다는 가정**이 있어야 한다는 것을 의미한다.

 도메인 적응의 기본적인 개념을 담고 있는 정리 1이 매우 어려운 내용이 아니라, 기본적인 공식들로 증명을 할 수 있다는 사실을 보여주기 위해 증명을 옮겨 써보겠다. 원문의 부록에 있는 증명보다 조금 더 세세하게 서술하였다.

> $$\begin{align}
> \epsilon_T(h) &= \epsilon_T(h) + \epsilon_S(h) - \epsilon_S(h) + \epsilon_S(h,f_T) - \epsilon_S(h,f_T) \\
&\le  \epsilon_S(h) + |\epsilon_S(h, f_T) - \epsilon_S(h, f_S)| + |\epsilon_T(h,f_T) - \epsilon_S(h,f_T)|\\
& = \epsilon_S(h) + |E_{\mathcal{D}_S}[|h(\mathbf{x}) - f_T(\mathbf{x})|] -E_{\mathcal{D}_S}[|h(\mathbf{x}) - f_S(\mathbf{x})|]+ |\epsilon_T(h,f_T) - \epsilon_S(h,f_T)| \\
&\le \epsilon_S(h) + E_{\mathcal{D}_S}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|] + |\epsilon_T(h,f_T) - \epsilon_S(h,f_T)| && \because |A|-|B| \le |A-B|\\
& = \epsilon_S(h) + E_{\mathcal{D}_S}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|] + |\int|h(\mathbf{x}) - f_T(\mathbf{x})|\phi_S(\mathbf{x})d\mathbf{x} - \int|h(\mathbf{x}) - f_T(\mathbf{x})|\phi_T(\mathbf{x})d\mathbf{x}| && \because \phi_{S,T}\text{ is density function of }\mathcal{D}_{S,T}\\
& \le \epsilon_S(h) + E_{\mathcal{D}_S}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|] + \int|h(\mathbf{x}) - f_T(\mathbf{x})||\phi_S(\mathbf{x})-\phi_T(\mathbf{x})|d\mathbf{x}\\
& \le \epsilon_S(h) + E_{\mathcal{D}_S}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|] + \int|\phi_S(\mathbf{x})-\phi_T(\mathbf{x})|d\mathbf{x} && \because |h(\mathbf{x})-f_T(\mathbf{x})| \le 1\\
& \le \epsilon_S(h) + E_{\mathcal{D}_S}[|f_S(\mathbf{x}) - f_T(\mathbf{x})|] + d(\mathcal{D}_S, \mathcal{D}_T)
\end{align}$$
>
> 여기서 맨 첫 번째 줄에서 $$\epsilon_S(h, f_T)$$ 대신 $$\epsilon_T(h, f_S)$$를 더하고 빼주면 $$\mathcal{D}_S$$대신 $$\mathcal{D}_T$$와 관계된 부등식을 얻을 수 있는데, 이 두 부등식을 한 번에 표현하면 정리 1과 같이 표현할 수 있다.

#### 정리2: $$\mathcal{H}$$-다이버전스를 이용한 상계

정리 1을 통해서 문제 1에 대한 대답을 개념적으로 얻었지만 실제로 데이터를 주었을 때 소스 도메인에서 학습한 분류기가 타겟 도메인에서 잘 동작하는지 추측하기 위해서는, 유한개의 샘플을 이용해서 우변에 해당하는 실제 타겟 오류의 상계를 구할 수 있어야 한다. 하지만 정리 1에서 등장하는 **$$L^1$$ 다이버전스(divergence)는 임의의 분포로부터의 유한개의 샘플을 이용해서 구할수 없음**이 알려져 있고, 또한 **$$L^1$$ 다이버전스는 지나치게 엄격**한 값이라서, 타겟 오류의 범위를 틀리지 않고 제한하려다보니 실제 오류값보다 훨씬 더 큰 값을 상계로 제시한다는 문제점이 있다. 

두 가지 문제 해결하기 위해 논문에서는 $$L^1$$ 다이버전스 대신 **$$\mathcal{H}$$-다이버전스**를 이용한다.

$$d_\mathcal{H}(\mathcal{D}, \mathcal{D'}) = 2\sup_{h\in\mathcal{H}}|\mathrm{Pr}_\mathcal{D}[h(\mathbf{x})=1] - \mathrm{Pr}_\mathcal{D'}[h(\mathbf{x})=1]|$$

$$\mathcal{H}$$-다이버전스는 L1 다이버전스에서처럼 모든 부분 집합을 확인하는 것이 아니라, 어떤 가설 집합 H에 속하는 가설로 주어지는 h를 기준으로 소스와 타겟 도메인에서 가장 결과가 엇갈리는 경우의 차이를 의미한다. **$$\mathcal{H}$$-다이버전스는 $$\mathcal{H}$$가 유한한 VC 차원(VC dimension, $$\mathcal{H}$$의 복잡도를 표현하는 수치)을 가진다면 유한개의 샘플로 추정할 수(논문의 보조 정리1, 보조 정리 2 참고) 있고, 또한 항상 L1 다이버전스보다 커질 수 없으므로** 앞에서 말한 두 가지 문제가 해결된다.

  타겟 오류의 상계를 얻기 위해서는 $$\mathcal{H}$$-다이버전스에서 $$\mathcal{H}$$ 가 특정 형태인 $$\mathcal{H}\Delta\mathcal{H}$$ 다이버전스를 사용하는데 대칭적인 차이 가설 공간 $$\mathcal{H}\Delta\mathcal{H}$$의 정의는 아래와 같다. $$\mathcal{H}\Delta\mathcal{H}$$는 $$\mathcal{H}$$에 속하는 어떤 가설들의 XOR 연산($$\oplus$$)을 새로운 가설로 하여 정의되는 공간으로 이해할 수 있다.  

$$g\in\mathcal{H}\Delta\mathcal{H}\Leftrightarrow g(\mathbf{x})=h(\mathbf{x})\oplus h'(\mathbf{x}) \text{ for some } h, h' \in \mathcal{H}$$
  
 
 그리하여 첫 번째 질문에 대해 유한개의 샘플로 추정할 수 있는 상계를 갖춘 답인 **정리 2**가 나오게 된다. 정리2도 역시 정리1처럼 보조정리들과 몇 가지 부등식을 이용하면 증명할 수 있다. 갑자기 XOR 연산과 관련된 $$\mathcal{H}\Delta\mathcal{H}$$-다이버전스가 왜 나왔는지 의문들었는데, 본 글에서는 생략한 증명 부분을 보면 일반적인 가설 $$h$$와 이상적인 가설 $$h^*$$의 관계를 이용해서 $$\epsilon_T(h)$$의 상계를 한정하는 과정에서 $$\mathcal{H}\Delta\mathcal{H}$$-다이버전스가 사용되는 것을 볼 수 있다.
> 정리 2
>
> $$\mathcal{H}$$이 VC 차원 $$d$$를 가지는 가설 공간이라고 하자. $$\mathcal{U}_S$$와 $$\mathcal{U}_T$$를 각각 $$\mathcal{D}_S$$와 $$\mathcal{D}_T$$에서 추출한 크기 $$m'$$의 레이블링이 되지않은 샘플들이라고 할 때,  어떤 $$\delta\in(0,1)$$와 모든 $$h\in\mathcal{H}$$ 에 대해 적어도 $$1-\delta$$확률로 다음을 만족한다.
>
> $$\epsilon_T(h)\le\epsilon_S(h) + \frac{1}{2}\hat{d}_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{U}_S, \mathcal{U}_T) + 4\sqrt{\frac{2d\log(2m') + \log(\frac{2}{\delta})}{m'}}+\lambda$$
>
> 여기서 $$d$$는 $$\mathcal{H}$$의 VC차원이며, $$\lambda$$= $$\min_{h\in\mathcal{H}}\epsilon_S(h)+\epsilon_T(h)$$이다.

정리2를 통해 정리 1과 유사하게 **타겟 오류**($$\epsilon_T(h)$$)는 **소스 오류**($$\epsilon_S(h)$$), **유한개의 샘플들을 이용해 구한 경험적인(empirical) 다이버전스**($$\hat{d}_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{U}_S, \mathcal{U}_T)$$), 그리고 **모델의 복잡도**($$d$$), **샘플 크기**($$m'$$), **소스와 타겟 오류의 합의 최솟값**($$\lambda$$)을 통해 상계를 정할 수 있다고 해석할 수 있고, 한 번 더 풀이하자면 소스와 타겟 모두 잘 작동하는 분류기가 존재한다는 가정하에서, **소스와 타겟 데이터 분포 사이의 거리가 적을 때, 복잡도가 적은 모델을 통해 학습한, 소스 오류가 낮은 분류기가 타겟도메인에서도 잘 동작할 것**이라는 암시를 얻을 수 있다.

### 두 번째 질문

이제 두 번째 질문에 대한 해답을 찾아보자. $$(1-\beta)*m$$개의 소스 데이터 포인트들을 가지고 있고, $$\beta*m$$개의 타겟 데이터 포인트들을 가지고 있다고 가정할 때, 이를 둘 다 활용해서 학습한다고 하면 도메인별로 로스를 따로 정의할 수 있고,  **최적화시 두 로스의 비중을 어떻게 주느냐**에 대한 문제로 두 번째 질문을 해석할 수 있다. 타겟 도메인의 로스에 대한 비중을 $$\alpha$$로 했을 때 최적화하고자 하는 전체 로스의 값을 식을 통해 표현하면 아래와 같다.

$$\hat{\epsilon}_\alpha(h) = \alpha*\hat{\epsilon}_T(h) +  (1-\alpha)\hat{\epsilon}_S(h)$$

여기서 $$\alpha$$가 1이 되면 타겟 데이터만을 이용해서 최적화를 진행하는 것이고, $$\alpha$$가 0이라면 소스 데이터만을 이용해서 학습한다는 의미이다.  단순히 균일한 비중($$\alpha=\frac{1}{2}$$)을 주는 것보다 나은 해답을 찾기 위해, 위 로스 함수를 통해 학습한 가설 $$h$$의 타겟 오류에 대하여 정리1 혹은 정리2와 유사하게 상계를 구하면 원문의 정리3이 되는데 정리3의 상계 값을 최소화하는 $$\alpha^j$$를 구하게 되면 아래와 같은 식이 나온다.


$$ \alpha^*(m_T, m_S: D) =
\begin{cases}
1 & m_T \ge D^2 \\
\min\{1,\nu\} & m_T \ge D^2
\end{cases}$$

$$ \nu = \frac{m_T}{m_T + m_S}(1+\frac{m_S}{\sqrt{D^2(m_S+m_T)-m_Sm_T}})$$

$$D=\sqrt{d}/A$$

$$A=\frac{1}{2}\hat{d}_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{U}_S,\mathcal{U}_T)+4\sqrt{\frac{2d\log(2m') + \log(\frac{4}{\delta})}{m'}}+\lambda$$

이를 해석해보면, 아래와 같은 네 가지 해석이 가능하다.
1. $$m_T$$가 0, 이면 $$\alpha^*$$는 0, $$m_S$$가 0이면 $$\alpha^*$$는 1이다. 즉 어떤 도메인의 데이터가 존재하지 않으면 데이터가 존재하는 도메인의 데이터만 이용해서 로스 함수를 구성할 수밖에 없다.
2. $$\hat{d}_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{U}_S,\mathcal{U}_T) = 0$$ 라면 $$A$$는 작아지고, $$D=\sqrt{d}/A$$는 커져서 $$\alpha^*$$는 $$\beta$$에 가까이 간다. 즉, 소스와 타겟 도메인이 같다고하면, 같은 도메인의 데이터로 생각하고 존재하는 데이터 크기대로 로스 함수의 가중치를 부여하면 된다.
3. $$m_T \ge D^2$$, 즉 타겟 데이터가 충분하다면, 소스 데이터를 이용해서 로스 함수를 구성하는 것은 타겟 도메인에서의 성능을 오히려 떨어뜨린다.
4. 타겟데이터가 적을 때도, 소스 데이터가 더 적다면($$m_S\lt\lt m_T$$) 소스데이터는 무시하는 편이 낫다. 따라서 **소스 데이터를 효과적으로 사용할 수 있는 상황은 타겟 데이터가 적고, 소스 데이터가 많은 상황** 뿐이다.


## 결론

해당 논문을 통해 도메인 적응 시나리오에서 **타겟 도메인에서의 오류의 상계를 레이블링이 없는 샘플들을 이용해서 구할 수 있음**을 확인하였고, 이를 통해 도메인 적응을 할 때 **소스 오류를 작게 하고, 소스와 타겟 데이터의 분포의 차이를 작게하면 타겟 오류도 작게 할 수 있을 것**이라는 현재 대부분의 도메인 적응 연구들이 취하고 있는 전략을 수식적으로 확인하였다. 또한 타겟 데이터가 일부 있을 때 이를 효과적으로 활용하기 위해서는 **소스와 타겟 분포의 차이**, **데이터양**에 따라 타겟 도메인에서의 로스함수에 대한 비중을 달리하여 주는 것이 좋음을 확인할 수 있었다.

위 내용 이외에 논문에는 해당 이론을 검증하기 위해 글의 긍정과 부정을 분류하는 문제를 대상으로 수행한 실험과 한발 더 나아가서 소스 도메인이 한 개가 아니라 여러 개인 경우에 대한 내용도 있으니, 더 관심 있으신 분들은 논문을 읽어보아도 좋을 것 같다. 
