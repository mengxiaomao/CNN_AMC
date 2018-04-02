function [prod]=calculate_pdf1(y,snr)
% M=[2,4,8,16];
M=[2,4,8,16,16,32,64];
N=length(y);
class_num=numel(M);

pdf_e=zeros(class_num,N);
prod=zeros(1,class_num);
for k=1:class_num
    xsym=1:M(k);
    xsym=xsym-1;
    x=scaling(k,xsym);
    N0=10^(-snr/10);
    x=x/sqrt(1+N0);
    N0=N0/(1+N0);
    
    for i=1:N
        p_e=0;
        for u=1:M(k)
            p_e=p_e+exp(-(abs(y(i)-x(u))^2)/N0);
%             p_e=p_e+exp((2*real(y(i)'*x(u))-abs(x(u))^2)/N0);
        end
        pdf_e(k,i)=p_e/M(k); 
    end
    prod(k)=sum(log(pdf_e(k,:)));  
end
prod=prod-max(prod);
prod=exp(prod);
scale=sum(prod);
prod=prod./scale;
end