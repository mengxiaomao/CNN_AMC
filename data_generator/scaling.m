function [y]=scaling(ind,xsym)
%1. 2PSK 2. 4PSK 3. 8PSK 4. 16QAM 5. 16APSK 6. 32APSK 7. 64QAM -1. Gaussian noise
M=[2,4,8,16,16,32,64];
if ind==1
    y=pskmod(xsym,M(ind));
elseif ind==2
    y=pskmod(xsym,M(ind));
elseif ind==3
    y=pskmod(xsym,M(ind));
elseif ind==4
    y=qammod(xsym,M(ind));
    y=y./sqrt(10);
elseif ind==5
    y=apskmod(xsym,M(ind));
    y=y./sqrt(1.0773);
elseif ind==6
    y=apskmod(xsym,M(ind));
    y=y./sqrt(2.4136);
elseif ind==7
    y=qammod(xsym,M(ind));
    y=y./sqrt(42);
elseif ind==-1
    Nsym=length(xsym);
    y=zeros(1,Nsym);
end
end