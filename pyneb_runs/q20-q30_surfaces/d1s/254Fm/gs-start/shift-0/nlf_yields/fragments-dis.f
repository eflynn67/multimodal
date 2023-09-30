      implicit real*8 (a-h,o-z)
      dimension nad(10000),nzd(10000),enr(10000),sn(10000)
      dimension kia(300),kiz(150),proa(81,300),proz(81,150),
     1          proc(81,300,150),prof(81,300),enpn(81,300,150)
      dimension kia1(300),kiz1(150) 
      dimension afmx(300),afmn(300),zfmx(150),zfmn(150)
      dimension afmx1(300),afmn1(300),zfmx1(150),zfmn1(150)
      dimension cfmx(300,150),cfs1(300,150),cfs2(300,150),zy(150)
      dimension ymx(300),ymn(300),yzx(150),yzn(150),caveh(300)
      dimension cavg(81,300),covg(81,300)
      dimension covmx(300),covmn(300),cavmx(300),cavmn(300)
      dimension avmx(300),avmn(300) 
      open(5,file='frag-loc.in',status='old')
      open(6,file='fragments.out',status='unknown')
      pi=4.0d0*datan(1.0d0)
      epl=0.000000001d0
      eps=0.0000001d0
      epp=0.01d0
      pr=1.0d0/3.0d0
      r0=1.12d0
      c1=sqrt(9.0d0/(20.0d0*pi))
      c2=3.0d0/(7.0d0*pi) 
      betl=0.0
      bets=0.0
      sigz=2.0d0
      sigm=3.0d0
      sigp=1.0d0
      sigb=0.5d0
      aven=40.0d0
      indx=0
      nern=1
      nerz=1
      nraa=35
      nrza=15
      
      read(5,*)nn11,nz11,nn21,nz21,nnr,nzr,exn,keyal
      
      aven=aven+exn
      nnt=nn11+nn21+nnr
      nzt=nz11+nz21+nzr
      nat=nnt+nzt
      nov=0
      kia=0
      kiz=0
      kia1=0
      kiz1=0
      proa=0.0d0
      proz=0.0d0
      proc=0.0d0
      amass=dfloat(nnt+nzt)
      do nn1=nn11-nern,nn11+nern
       do nz1=nz11-nerz,nz11+nerz
        do nn2=nn21-nern,nn21+nern
         do nz2=nz21-nerz,nz21+nerz
          nov=nov+1
          nn=nnt-nn1-nn2
          nz=nzt-nz1-nz2 
          emin=10000000.0d0
          emin1=-100000.0d0
          nct=0
          nad=0
          nzd=0
          enr=0.0d0

          do in=0,nn
           do iz=0,nz

            nct=nct+1

            nn1f=nn1+in
            nz1f=nz1+iz
            nn2f=nnt-nn1f
            nz2f=nzt-nz1f  
            na1f=nn1f+nz1f
            na2f=nn2f+nz2f
        
            if(na1f.gt.na2f)then
             nad(nct)=na1f
            else
             nad(nct)=na2f
            endif
            if(nz1f.gt.nz2f)then
             nzd(nct)=nz1f
            else
             nzd(nct)=nz2f
            endif
 
            r1=r0*dfloat(na1f)**pr
            r2=r0*dfloat(na2f)**pr
            rs=r1+r2
            t1=1.44*dfloat(nz1f*nz2f)/rs
c############## correction in Coulomb ###########
            if(na1f.gt.na2f)then
             bet1=betl
             bet2=bets
            else
             bet1=bets
             bet2=betl
            endif
            v1=c1*bet1*r1**2+c2*(bet1*r1)**2
            v2=c1*bet2*r2**2+c2*(bet2*r2)**2
            coeff=v1+v2
            t2=nz1f*nz2f*coeff*1.44/(rs**3.0d0)
c################################################
            coul=t1+t2
            if(indx.eq.0)then 
             call amassff(na1f,nz1f,bet1,def1)
             call amassff(na2f,nz2f,bet2,def2)
            else
             call amassld(indx,na1f,nz1f,def1)
             call amassld(indx,na2f,nz2f,def2)
            endif

            enr(nct)=coul+def1+def2
            if(coul+def1+def2.lt.emin)then
             emin=coul+def1+def2
             naf=na2f
             nzf=nz2f
             mic=nct
            endif
           enddo
          enddo
          sn=0.0d0 
          do i=1,nct
           e=emin-enr(i)+aven
           if(e.gt.0.0)then
            am1=dfloat(nad(i))
            am2=dfloat(nnt+nzt-nad(i))
            al1=am1/10.0d0
            al2=am2/10.0d0
c#####################################################
            p1=5.d0/3.d0
            p2=3.d0/2.d0
            p3=9.d0/4.d0
            p4=1.d0/2.d0
            p5=5.d0/2.d0
            arg=2.0d0*((al1+al2)*e)**p4
            fac1=(am1**p1*am2**p1/(am1**p1+am2**p1))**p2
            fac2=(am1*am2/(am1+am2))**p2
            fac3=(al1*al2)**p4/(al1+al2)**p5
            fac4=(1.0d0-(1.0d0/(2.d0*((al1+al2)*e)**p4)))*e**p3
c#######################################################
c            p1=5.d0/3.d0
c            p2=3.d0/2.d0
c            p3=9.d0/4.d0
c            p4=1.d0/2.d0
c            p5=11.d0/4.d0
c            arg=2.0d0*((al1+al2)*e)**p4
c            fac1=(am1**p1*am2**p1/(am1**p1+am2**p1))**p2
c            fac2=(am1*am2/(am1+am2))**p2
c            fac3=(al1*al2)**p4/(al1+al2)**p5
c            fac4=(1.0d0-(19.0d0/(8.d0*((al1+al2)*e)**p4)))*e**p3
c#######################################################
c            p1=5.d0/3.d0
c            p2=3.d0/2.d0
c            p3=10.d0/4.d0
c            p4=1.d0/2.d0
c            p5=12.d0/4.d0
c            arg=2.0d0*((al1+al2)*e)**p4
c            fac1=(am1**p1*am2**p1/(am1**p1+am2**p1))**p2
c            fac2=(am1*am2/(am1+am2))**p2
c            fac3=(al1*al2)**p4/(al1+al2)**p5
c            fac4=(1.0d0-(7.0d0/(2.d0*((al1+al2)*e)**p4)))*
c     1            (dfloat(nzd(i)*(nzt-nzd(i)))**p4)*e**p3
c########################################################
            sn(i)=fac1*fac2*fac3*fac4*dexp(arg)
           endif      
          enddo
          
          do i=1,nct
           ia=nad(i)            
           iz=nzd(i)
           proc(nov,ia,iz)=sn(i)/sn(mic)
           enpn(nov,ia,iz)=emin-enr(i)+aven
           if(proc(nov,ia,iz).lt.epl)then
            proc(nov,ia,iz)=0.0d0
            enpn(nov,ia,iz)=0.0d0
           endif 
           proc(nov,nat-ia,nzt-iz)=proc(nov,ia,iz)
           enpn(nov,nat-ia,nzt-iz)=enpn(nov,ia,iz)
          enddo 

          nah=nat/2
          ika=0
          do ia=nah,nat
           ika=ika+1
           pro=0.0d0
           do j=1,nct
            if(nad(j).eq.ia)then
             pro=pro+sn(j)/sn(mic)
            endif
           enddo
           if(pro.lt.eps)pro=0.0d0
           kia(ika)=ia
           proa(nov,ika)=pro 
          enddo
          nzh=nzt/2
          ikb=0
          do iz=nzh,nzt
           ikb=ikb+1
           pro=0.0d0
           do j=1,nct
            if(nzd(j).eq.iz)then
             pro=pro+sn(j)/sn(mic)
            endif
           enddo
           if(pro.lt.eps)pro=0.0d0
           kiz(ikb)=iz
           proz(nov,ikb)=pro
          enddo       
         enddo
        enddo
       enddo
      enddo

c################### raw distributions ######################      
      do i=1,ika
       afmx(i)=proa(1,i)
       afmn(i)=proa(1,i)
       do j=2,nov
        if(proa(j,i).gt.afmx(i))afmx(i)=proa(j,i)
        if(proa(j,i).lt.afmn(i))afmn(i)=proa(j,i)
       enddo 
      enddo
c--------------------for sym distbn----------------------------
      if(nn11.eq.nn21.and.nz11.eq.nz21)then
       nwa=0 
       do i=ika,1,-1
        nwa=nwa+1
        afmx1(nwa)=afmx(i)
        afmn1(nwa)=afmn(i)
        kia1(nwa)=nnt+nzt-kia(i)
       enddo 
       do i=2,ika
        nwa=nwa+1
        afmx1(nwa)=afmx(i)
        afmn1(nwa)=afmn(i)
        kia1(nwa)=kia(i)
       enddo
       afmx=afmx1
       afmn=afmn1
       kia=kia1
       ika=nwa
      endif      
c---------------------------------------------------------------              
      do i=1,ikb
       zfmx(i)=proz(1,i)
       zfmn(i)=proz(1,i)
       do j=2,nov
        if(proz(j,i).gt.zfmx(i))zfmx(i)=proz(j,i)
        if(proz(j,i).lt.zfmn(i))zfmn(i)=proz(j,i)
       enddo 
      enddo
c---------------------for sym distbn-----------------------------
      if(nn11.eq.nn21.and.nz11.eq.nz21)then
       nwb=0 
       do i=ikb,1,-1
        nwb=nwb+1
        zfmx1(nwb)=zfmx(i)
        zfmn1(nwb)=zfmn(i)
        kiz1(nwb)=nzt-kiz(i)
       enddo 
       do i=2,ikb
        nwb=nwb+1
        zfmx1(nwb)=zfmx(i)
        zfmn1(nwb)=zfmn(i)
        kiz1(nwb)=kiz(i)
       enddo
       zfmx=zfmx1
       zfmn=zfmn1
       kiz=kiz1
       ikb=nwb
      endif
c--------------------------------------------------------------              
      do ia=1,nat
       do iz=1,nzt
        cfmx(ia,iz)=proc(1,ia,iz)
        do i=2,nov
         if(proc(i,ia,iz).gt.cfmx(ia,iz))cfmx(ia,iz)=proc(i,ia,iz)
        enddo 
       enddo 
      enddo

c########### N-Z correlation ################################ 
      if(keyal.ne.0)then
       sig=1.0d0
       cfs1=0.0d0
       do i=nah-nraa,nah+nraa
        do j=nzh-nrza,nzh+nrza,2
         ya=0.0d0
         do k=nzh-nrza,nzh+nrza,2
          y1=(j-k+0.5d0)/(sqrt(2.d0)*sig)
          y2=(j-k-0.5d0)/(sqrt(2.d0)*sig)
          call erf(y1,v3)
          call erf(y2,v4)
          ya1=cfmx(i,k)*(v3-v4)
          cfs1(i,j)=cfs1(i,j)+ya1
         enddo
        enddo
       enddo
       do i=nah-nraa,nah+nraa
        do j=nzh-nrza+1,nzh+nrza,2
         ya=0.0d0
         do k=nzh-nrza+1,nzh+nrza,2
          y1=(j-k+0.5d0)/(sqrt(2.d0)*sig)
          y2=(j-k-0.5d0)/(sqrt(2.d0)*sig)
          call erf(y1,v3)
          call erf(y2,v4)
          ya1=cfmx(i,k)*(v3-v4)
          cfs1(i,j)=cfs1(i,j)+ya1
         enddo
        enddo
       enddo

       cfs2=0.0d0
       ytot=0.0d0
       do i=nah,nah+nraa
        do j=nzh,nzh+nrza
         ya=0.0d0
         do l=nah,nah+nraa
          do k=nzh,nzh+nrza
           x1=(i-l+0.5d0)/(sqrt(2.d0)*3.0)
           x2=(i-l-0.5d0)/(sqrt(2.d0)*3.0)
           call erf(x1,v1)
           call erf(x2,v2)
           y1=(j-k+0.5d0)/(sqrt(2.d0)*2.0)
           y2=(j-k-0.5d0)/(sqrt(2.d0)*2.0)
           call erf(y1,v3)
           call erf(y2,v4)
           ya1=cfs1(l,k)*(v1-v2)*(v3-v4)
           cfs2(i,j)=cfs2(i,j)+ya1
          enddo
         enddo
         ytot=ytot+cfs2(i,j)
        enddo
       enddo
       write(6,*)'N-Z distribution'
       write(6,*)'Z-----N-----A----probability'
       zy=0.0d0
       cfs1=0.0d0
       do j=nzh,nzh+nrza
        yyy=0.0d0
        do i=nah,nah+nraa
         quan=(cfs2(i,j)/ytot)*100.0d0
         cfs1(i,j)=quan
         if(quan.gt.0.00001d0)write(6,*)j,i-j,i,quan
        enddo
       enddo
       write(6,*)'charge polarization for heavy fragments'
       write(6,*)'A----------polarization'
       do i=nah,nah+nraa
        zav=0.0d0
        tot=0.0d0
        key=0
        do j=nzh,nzh+nrza
         if(cfs1(i-1,j).gt.0.01d0)then
          zav=zav+j*cfs1(i-1,j)
          tot=tot+cfs1(i-1,j)
          key=1
         endif        
        enddo
        if(key.gt.0)then
         chp=zav/tot-dfloat(nzt*i)/dfloat(nat)
        else
         chp=0.0d0
        endif 
        write(6,*)i,chp
       enddo
      endif
c################# mass smoothening ####################
      write(6,*)'gaussian weighted mass'   
      ymx=0.0d0
      ymn=0.0d0
      do i=1,ika
       yamx=0.0d0
       yamn=0.0d0
       do j=1,ika
        x1=(kia(i)-kia(j)+0.5d0)/(sqrt(2.d0)*sigm)
        x2=(kia(i)-kia(j)-0.5d0)/(sqrt(2.d0)*sigm)
        call erf(x1,v1)
        call erf(x2,v2)
        ya1=afmx(j)*(v1-v2)
        ya2=afmn(j)*(v1-v2)
        yamx=yamx+ya1
        yamn=yamn+ya2
       enddo
       ymx(i)=yamx  
       ymn(i)=yamn  
      enddo
     
      valu1=0.0d0
      valu2=0.0d0
      do i=1,ika
       valu1=valu1+ymx(i)
       valu2=valu2+ymn(i)
      enddo 

      if(nn11.eq.nn21.and.nz11.eq.nz21)then
       do i=1,ika
        am1=(ymx(i)/valu1)*200.0d0
        am2=(ymn(i)/valu2)*200.0d0
        if(am1.gt.epp.or.am2.gt.epp)write(6,13)kia(i),am1
       enddo
       do i=1,ika
        am1=(ymx(i)/valu1)*200.0d0
        am2=(ymn(i)/valu2)*200.0d0
        mas=nat-kia(i)
        if(am1.gt.epp.or.am2.gt.epp)write(6,13)mas,am2
       enddo
      else
       do i=ika,2,-1
        am1=(ymx(i)/valu1)*100.0d0
        am2=(ymn(i)/valu2)*100.0d0
        mas=nat-kia(i)
        if(am1.gt.epp.or.am2.gt.epp)write(6,13)mas,am1
       enddo
       do i=1,ika
        am1=(ymx(i)/valu1)*100.0d0
        am2=(ymn(i)/valu2)*100.0d0
        if(am1.gt.epp.or.am2.gt.epp)write(6,13)kia(i),am1
       enddo

       do i=ika,1,-1
        am1=(ymx(i)/valu1)*100.0d0
        am2=(ymn(i)/valu2)*100.0d0
        if(am1.gt.epp.or.am2.gt.epp)write(6,13)kia(i),am2
       enddo
       do i=2,ika
        am1=(ymx(i)/valu1)*100.0d0
        am2=(ymn(i)/valu2)*100.0d0
        mas=nat-kia(i)
        if(am1.gt.epp.or.am2.gt.epp)write(6,13)mas,am2
       enddo
      endif
c##################### charge smoothening ###############################
      write(6,*)'gaussian weighted charge'
      if((ikb/2)*2.eq.ikb)then
       ike=ikb
       iko=ikb-1
      else
       ike=ikb-1
       iko=ikb       
      endif
c-------odd loop---------------------
      do i=1,iko,2
       yzmx=0.0d0
       yzmn=0.0d0
       do j=1,iko,2
        x1=(kiz(i)-kiz(j)+0.5d0)/(sqrt(2.d0)*sigz)
        x2=(kiz(i)-kiz(j)-0.5d0)/(sqrt(2.d0)*sigz)
        call erf(x1,v1)
        call erf(x2,v2)
        ya1=zfmx(j)*(v1-v2)
        ya2=zfmn(j)*(v1-v2)
        yzmx=yzmx+ya1
        yzmn=yzmn+ya2
       enddo
       yzx(i)=yzmx
       yzn(i)=yzmn
      enddo
c-------even loop---------------------
      do i=2,ike,2
       yzmx=0.0d0
       yzmn=0.0d0
       do j=2,ike,2
        x1=(kiz(i)-kiz(j)+0.5d0)/(sqrt(2.d0)*sigz)
        x2=(kiz(i)-kiz(j)-0.5d0)/(sqrt(2.d0)*sigz)
        call erf(x1,v1)
        call erf(x2,v2)
        ya1=zfmx(j)*(v1-v2)
        ya2=zfmn(j)*(v1-v2)
        yzmx=yzmx+ya1
        yzmn=yzmn+ya2
       enddo
       yzx(i)=yzmx
       yzn(i)=yzmn
      enddo

      zfmx=yzx
      zfmn=yzn
      yzx=0.0d0
      yzn=0.0d0

      do i=1,ikb
       yzmx=0.0d0
       yzmn=0.0d0
       do j=1,ikb
        x1=(kiz(i)-kiz(j)+0.5d0)/(sqrt(2.d0)*sigb)
        x2=(kiz(i)-kiz(j)-0.5d0)/(sqrt(2.d0)*sigb)
        call erf(x1,v1)
        call erf(x2,v2)
        ya1=zfmx(j)*(v1-v2)
        ya2=zfmn(j)*(v1-v2)
        yzmx=yzmx+ya1
        yzmn=yzmn+ya2
       enddo
       yzx(i)=yzmx
       yzn(i)=yzmn
      enddo

      valu1=0.0d0
      valu2=0.0d0
      do i=1,ikb
       valu1=valu1+yzx(i)
       valu2=valu2+yzn(i)
      enddo
      
      if(nn11.eq.nn21.and.nz11.eq.nz21)then 
       do i=1,ikb
        az1=(yzx(i)/valu1)*200.0d0
        az2=(yzn(i)/valu2)*200.0d0
        if(az1.gt.epp.or.az2.gt.epp)write(6,13)kiz(i),az1
       enddo 
       do i=2,ikb
        az1=(yzx(i)/valu1)*200.0d0
        az2=(yzn(i)/valu2)*200.0d0
        mzs=nzt-kiz(i)
        if(az1.gt.epp.or.az2.gt.epp)write(6,13)mzs,az2
       enddo 
      else
       do i=ikb,2,-1
        az1=(yzx(i)/valu1)*100.0d0
        az2=(yzn(i)/valu2)*100.0d0
        mzs=nzt-kiz(i)
        if(az1.gt.epp.or.az2.gt.epp)write(6,13)mzs,az1
       enddo      
       do i=1,ikb
        az1=(yzx(i)/valu1)*100.0d0
        az2=(yzn(i)/valu2)*100.0d0
        if(az1.gt.epp.or.az2.gt.epp)write(6,13)kiz(i),az1
       enddo 

       do i=ikb,1,-1
        az1=(yzx(i)/valu1)*100.0d0
        az2=(yzn(i)/valu2)*100.0d0
        if(az1.gt.epp.or.az2.gt.epp)write(6,13)kiz(i),az2
       enddo 
       do i=2,ikb
        az1=(yzx(i)/valu1)*100.0d0
        az2=(yzn(i)/valu2)*100.0d0
        mzs=nzt-kiz(i)
        if(az1.gt.epp.or.az2.gt.epp)write(6,13)mzs,az2
       enddo
      endif 
      if(keyal.eq.0)then
       close(6)
       stop
      endif
c####################################################################
      write(6,*)'odd-even difference in Z (%)'
      evzm=0.0d0
      evzn=0.0d0
      odzm=0.0d0
      odzn=0.0d0

      do i=1,ikb
       p1=(yzx(i)/valu1)*100.0d0
       p2=(yzn(i)/valu2)*100.0d0
       if((kiz(i)/2)*2.eq.kiz(i))then
        if(p1.gt.p2)then
         evzm=evzm+p1
         evzn=evzn+p2
        else
         evzm=evzm+p2
         evzn=evzn+p1
        endif
       else
        if(p1.gt.p2)then
         odzm=odzm+p1
         odzn=odzn+p2
        else
         odzm=odzm+p2
         odzn=odzn+p1
        endif
       endif
      enddo
      del1=100.0d0*(evzm-odzn)/(evzm+odzn)
      del2=100.0d0*(evzn-odzm)/(evzn+odzm) 
      write(6,*)del1,del2
c######################## partial charge smoothening ###################      
      write(6,*)'partial mass distributions for different Z after 
     1n-evaporation'
      cavg=0.0d0
      covg=0.0d0
      beta=0.0d0
      prof=0.0d0
      extt=aven
      nevn=0
      nevp=0
      do iz=1,nzt
       covmx=0.0d0
       covmn=0.0d0
       cavmx=0.0d0
       cavmn=0.0d0
       ymx  =0.0d0
       ymn  =0.0d0
       key  =0

       do ii=1,nov
        valt=0.0d0
        do ia=1,nat
         valt=valt+proc(ii,ia,iz)
        enddo
        if(valt.lt.epl)cycle
        key=1
        caveh=0.0d0
        do ia=2,nat
         if(proc(ii,ia,iz).lt.epl)cycle
         na=ia
         nevn=nevn+1
c         ex=extt*dfloat(na)/dfloat(nat)
         ex=enpn(ii,ia,iz)*dfloat(na)/dfloat(nat)
         kn=0
         do
          call amassff(na,iz,beta,ben1)
          call amassff(na-1,iz,beta,ben2)
          temp=dsqrt(ex*10.0d0/dfloat(na))
          bbn=dabs(ben1-ben2)
          ex=ex-(bbn+temp)
          if(ex.gt.0.0d0)then
           nevp=nevp+1
           kn=kn+1
           na=na-1
          else
           exit
          endif
         enddo
         if(kn.lt.ia)caveh(ia-kn)=caveh(ia-kn)+proc(ii,ia,iz)
        enddo  
        do ia=1,nat 
         prof(ii,ia)=prof(ii,ia)+caveh(ia) 
         cavg(ii,ia)=caveh(ia)
         covg(ii,ia)=proc(ii,ia,iz)
        enddo
       enddo
       if(key.eq.0)cycle

       do i=1,nat
        covmx(i)=covg(1,i)
        covmn(i)=covg(1,i)
        cavmx(i)=cavg(1,i)
        cavmn(i)=cavg(1,i)
        do j=2,nov
         if(cavg(j,i).gt.cavmx(i))cavmx(i)=cavg(j,i)
         if(cavg(j,i).le.cavmn(i))cavmn(i)=cavg(j,i)
         if(covg(j,i).gt.covmx(i))covmx(i)=covg(j,i)
         if(covg(j,i).le.covmn(i))covmn(i)=covg(j,i)
        enddo
       enddo
       do i=30,nat
        yamx=0.0d0
        yamn=0.0d0
        do j=30,nat
         x1=(i-j+0.5d0)/(sqrt(2.d0)*sigp)
         x2=(i-j-0.5d0)/(sqrt(2.d0)*sigp)
         call erf(x1,v1)
         call erf(x2,v2)
         ya1=cavmx(j)*(v1-v2)
         ya2=cavmn(j)*(v1-v2)
         yamx=yamx+ya1
         yamn=yamn+ya2
        enddo
        ymx(i)=yamx  
        ymn(i)=yamn  
       enddo
     
       val1=0.0d0
       val2=0.0d0
       val3=0.0d0
       val4=0.0d0
       val5=0.0d0
       val6=0.0d0
       do i=1,nat
        val1=val1+ymx(i)
        val2=val2+ymn(i)
        val3=val3+covmx(i)
        val4=val4+covmn(i)
        val5=val5+cavmx(i)
        val6=val6+cavmn(i)
       enddo
       write(6,*)'Z = (smoothen after evap-----original-----after evap)
     1 ',iz
       do i=1,nat
        am1=(ymx(i)/val1)*100.0d0
        if(val2.gt.epl)then
         am2=(ymn(i)/val2)*100.0d0
        else
         am2=am1
        endif
        ao1=(covmx(i)/val3)*100.0d0
        if(val4.gt.epl)then
         ao2=(covmn(i)/val4)*100.0d0
        else
         ao2=ao1
        endif
        an1=(cavmx(i)/val5)*100.0d0
        if(val6.gt.epl)then
         an2=(cavmn(i)/val6)*100.0d0
        else
         an2=an1
        endif
 
        if(am1.gt.epp.or.am2.gt.epp.or.ao1.gt.epp.or.ao2.gt.epp)
     1  write(6,13)i,am1,am2,ao1,ao2,an1,an2
       enddo
      enddo
      write(6,*)'evaporated neutron multiplicity'
       write(6,*)2.0d0*dfloat(nevp)/dfloat(nevn)
c############### mass distribution after n-evaporation ##############
      avmx=0.0d0
      avmn=0.0d0
      do i=1,nat
       avmx(i)=prof(1,i)
       avmn(i)=prof(1,i)
       do j=2,nov
        if(prof(j,i).gt.avmx(i))avmx(i)=prof(j,i)
        if(prof(j,i).le.avmn(i))avmn(i)=prof(j,i)
       enddo
      enddo

      ymx=0.0d0
      ymn=0.0d0
      do i=30,nat
       yamx=0.0d0
       yamn=0.0d0
       do j=30,nat
        x1=(i-j+0.5d0)/(sqrt(2.d0)*sigm)
        x2=(i-j-0.5d0)/(sqrt(2.d0)*sigm)
        call erf(x1,v1)
        call erf(x2,v2)
        ya1=avmx(j)*(v1-v2)
        ya2=avmn(j)*(v1-v2)
        yamx=yamx+ya1
        yamn=yamn+ya2
       enddo
       ymx(i)=yamx  
       ymn(i)=yamn  
      enddo
      
      val1=0.0d0
      val2=0.0d0
      do i=1,nat
       val1=val1+ymx(i)
       val2=val2+ymn(i)
      enddo
      write(6,*)'-----secondary mass distribution-----'
      do i=1,nat
       am1=(ymx(i)/val1)*200.0d0
       am2=(ymn(i)/val2)*200.0d0
       mas=i
       if(am1.gt.epp.or.am2.gt.epp)
     1  write(6,13)mas,am1,am2
      enddo

      close(5)
      close(6)

 13   format(i6,6f12.4)     
      stop
      end
c################################################################
c                END MAIN PROGRAM
c################################################################

c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      subroutine amassff(na,nz,beta,defn)
      implicit real*8 (a-h,o-z)
c----  subroutine to  calculate the b.e. (a,z) using the
c----  global liquid-drop parameters of Myers and Swiatecki,
c----  Nucl.Phys.81(1966)1
      
      ap=dfloat(na)
      zp=dfloat(nz)
      a1=15.677
      a2=18.56
      c3=0.717
      c4=1.21129
      ak=1.79
      del=11.0
      spi=(ap-2.0*zp)/ap
      
      c1=a1*(1.0-ak*spi**2)
      c2=a2*(1.0-ak*spi**2)
      d1=(1.0+(2.0/5.0)*beta**2)
      d2=(1.0-(1.0/5.0)*beta**2)

      ia=nint(ap)
      iz=nint(zp)
      a13=ap**(1./3.)

      delta1=4.31d0/ap**(0.31d0)
c      delta1=11.0/ap**(0.5d0) 
      if((ia-2*(ia/2)).eq.0) then
       if((iz-2*(iz/2)).eq.0) then
         delta=-delta1
       else
         delta=delta1
       endif
      else
       delta=0.d0
      endif

      defn=-c1*ap+c2*a13*a13*d1+c3*zp*zp*d2/a13-c4*zp*zp/ap+delta
      return
      end subroutine

c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

      subroutine amassld(indx,na,nz,defn)
      implicit real*8 (a-h,o-z)
      character as
c---subroutine to  calculate the experimental b.e. of a nucleus (a,z)
      ap=dfloat(na)
      zp=dfloat(nz)
      defn1=8.071
      defp=7.289
      def=0.0d0
      if(indx.eq.1)then
       open(10,file='mass-table.in',status='old') 
       do ii=1,9107 
        read(10,*)iz,inn,ia,sh,df 
        if(iz.eq.nz.and.ia.eq.na)then
         def=df
         exit
        endif
       enddo
       close(10)
       defn=-defp*zp-defn1*(ap-zp)+def
      elseif(indx.eq.2)then    
c       open(10,file='mass-table-unedf1.txt',status='old') 
c       do ii=1,8324 
       open(10,file='mass-table-skm.txt',status='old') 
       do ii=1,8700 
        read(10,*)as,iz,inn,ia,df 
        if(iz.eq.nz.and.ia.eq.na)then
         def=df
         exit
        endif
       enddo
       close(10)
       defn=def
      endif 
      return
      end subroutine    
    
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

      subroutine erf(xx,valu1)
c-- calculates the error function
      implicit real*8 (a-h,o-z)
      dimension f(4001)
      nn=4000
      nn1=4001
      dx=xx/dfloat(nn)
      do i=1,nn+1
        arg=dfloat(i-1)*dx
        f(i)=dexp(-arg*arg)
      enddo
      call simp(f,dx,nn1,valu)
      pi=4.0*datan(1.0d0)
      valu1=valu/dsqrt(pi)
      return
      end subroutine

c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

      subroutine simp(g,h,m,valu)
c-----INTEGRATE FUNCTION g(z) using (m) equispaced points with step h
      implicit real*8 (a-h,o-z)
      dimension g(m)
      n=m-1
      sum1=0.d0
      sum2=0.d0
      if(n.eq.2)then
       valu=(h/3.d0)*(g(1)+4.d0*g(2)+g(3))
       return
      endif
      do i=2,n,2
       sum1=sum1+g(i)
      enddo
      do j=3,n-1,2
       sum2=sum2+g(j)
      enddo
      valu=(h/3.d0)*(g(1)+4.d0*sum1+2.d0*sum2+g(n+1))
      return
      end subroutine  
 
c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      
      subroutine wncn(acn1,zcn1,ex,bbn,width)
c----calculation of pre-scission  neutron width in MeV for CN----
      implicit real*8 (a-h,o-z)
      dimension fy(20001)
      nnt=1000

      call pair(acn1      ,zcn1,del1)
      call pair(acn1-1.0d0,zcn1,del2)
      beta=0.0d0
      nz=nint(zcn1)
      na=nint(acn1)
      call amassff(na,nz,beta,def1)
      call amassff(na-1,nz,beta,def2)
      bbn=def2-def1
      epmax=ex+def1-def2
      if(epmax.lt.0.0d0)then
       width=0.0d0
       return
      endif 
      emaxd=epmax
      dele=epmax/dfloat(nnt)

      a1=acn1/10.0d0
      arg1=2.0d0*dsqrt(a1*(ex-del1))

      acn2=acn1-1.0d0
      a2=acn2/10.0d0
      t2=acn2**(0.3333333d0)
      rad=1.21d0*(t2+1.0d0)  
         
      do i=2,nnt+1
       erun=dfloat(i-1)*dele
       earg=emaxd-erun
       if(earg.le.0.0001d0.or.earg.le.del2) then
        fy(i)=0.0d0
        cycle
       endif
       arg2=2.0d0*dsqrt(a2*(earg-del2))
       arg=arg2-arg1
       fy(i)=erun*dexp(arg)/(earg-del2)**2
       fy(i)=fy(i)*dsqrt(a2/a1)
       rade=rad+3.4d0/dsqrt(erun)
       siginv=rade**2
       fy(i)=fy(i)*siginv
      enddo
      fy(1)=0.0
      call simp(fy,dele,nnt,valu)
      width=(ex-del1)**2*valu
      return
      end subroutine

c@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 
      subroutine pair(a,z,delta)
      implicit real*8 (a-h,o-z)
!------calculates pairing energy for a nucleus (a,z)-------------------
      ap=a
      zp=z
      del=11.0
      ia=ap+0.0001
      iz=zp+0.0001
      if((ia-2*(ia/2)).eq.0) then
       if((iz-2*(iz/2)).eq.0) then
        delta= del/sqrt(ap)
       else
        delta=-del/sqrt(ap)
       endif
      else
       delta=0.
      endif
      return
      end subroutine     
