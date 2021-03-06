function [co]=cumulant(Y)
[N,~]=size(Y);
co=zeros(N,10);
for i=1:N
    y=Y(i,:);
    cy=conj(y);
    M20=moment1(y,cy,2,0);
    M21=moment1(y,cy,2,1);
    M22=moment1(y,cy,2,2);
    M40=moment1(y,cy,4,0);
    M41=moment1(y,cy,4,1);
    M42=moment1(y,cy,4,2);
    M43=moment1(y,cy,4,3);
    M60=moment1(y,cy,6,0);
    M61=moment1(y,cy,6,1);
    M62=moment1(y,cy,6,2);
    M63=moment1(y,cy,6,3);
    M80=moment1(y,cy,8,0);
    C20=M20;
    C21=M21;
    C40=M40-3*M20^2;
    C41=M41-3*M20*M21;
    C42=M42-abs(M20)^2-2*M21^2;
    C60=M60-15*M40*M20+30*M20^3;
    C61=M61-5*M40*M21-10*M41*M20+30*M20^2*M21;
    C62=M62-6*M42*M20-8*M41*M21-M40*M22+6*M20^2*M22+24*M21^2*M20;
    C63=M63-9*M42*M21+12*M21^3-3*M43*M20-3*M41*M22+18*M22*M21*M20;
    C80=M80-28*M60*M20-35*M40^2+420*M40*M20^2-630*M20^4;
    co(i,:)=[C20,C21,C40^(1/2),C41^(1/2),C42^(1/2),C60^(1/3),C61^(1/3),C62^(1/3),C63^(1/3),C80^(1/4)];
end
% co=abs(co);
co=[real(co),imag(co)];
end