function [gamma]=calculate_pdf_noncoherent3(y,snr)
%  Simpson's rules
M=[2,4,8,16];
window=90;
dots=45;
class_num=numel(M);

theta=linspace(-window,window,dots)/180*pi;
prod=zeros(numel(theta),class_num);
for v=1:numel(theta)
    
    y1=y.*exp(sqrt(-1)*theta(v));
    prod(v,:)=calculate_pdf2(y1,snr);
end
maxprod=max(prod(:));
gamma=prod-maxprod;
gamma=exp(gamma);
gamma=sum(gamma,1);
gamma=gamma./sum(gamma);
end