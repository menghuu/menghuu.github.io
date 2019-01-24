---
title: building_ldapd_environment
tags: 
    - linux 
    - docker
    - openldap
    - apacheds
    - phpldapadmin
---

# 搭建 ldapd 认证环境

## openldap 还是 apacheds ?

说实话，还是apacheds(Apache Directory Server)带的那个Apache Directory Studio感觉比 openldap + phpldapadmin 要舒服的多。事实上是可以配合着openldap-server使用apache directory studio，链接在这里[^using_apached_directory_studio_with_openldapd]

不过从参考资料来看，openldapd的配置可能更多一些，虽然本文使用的是docker，可能配置起来没有那么难，但是为了保险还是使用了openldapd + phpldapadmin / apache_directory_studio。

## docker 是什么?

> Docker 是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。

简单来说，就是将依赖环境打包起来的一个工具。docker的一个特点就是，默认一个docker镜像启动之后，如果关闭则所有数据都消失了，但是创建/启动起来异常的快，比虚拟机快得多了。具体的一些原理这里不多说了

## docker-compose 是什么

由于docker的每个容器目标都是一个应用，如果是由多个docker容器我们如何将它们配合起来？说实话，有很多种的方法，这里采用最简单的docker-compose工具来提供这样的编排。

## 步骤

### 安装docker

略，大概就是`sudo apt install docker.io`，如果你不想用root权限执行docker命令，那么可以将自己的用户加到docker这组中。

### 安装docker-compose

> 官方的安装教程: https://docs.docker.com/compose/install/

简单来说

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.23.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose  
sudo chmod +x /usr/local/bin/docker-compose  
docker-compose --version
```

### 创建docker-compose.yml

主要是配置`openldap`，docker版的tutorial在[这里](https://github.com/osixia/docker-openldap#quick-start),以及docker版的`phpldapadmin`这[这里](https://github.com/osixia/docker-phpLDAPadmin)

```bash
mkdir -p $HOME/projs/lab_docker_compose/ldap_persistence/ldap # 存放database
mkdir -p $HOME/projs/lab_docker_compose/ldap_persistence/slapd.d # 存放config文件
mkdir -p $HOME/projs/lab_docker_compose/ldap_persistence/backup # 用来备份
```

**无论你看到什么别的教程，能不修改slapd.conf就不修改这个文件，这种配置方案已经不再被鼓励**

```yml
version: '3'
networks:
  ldap_service_network:
services:
  ldap-service:
    image: 'osixia/openldap-backup:1.1.8' # 这里使用back-up的这个，能自动备份
    #build: ./docker-openldap-backup/image
    volumes:
      - './ldap-persistence/ldap:/var/lib/ldap'
      - './ldap-persistence/slapd.d:/etc/ldap/slapd.d'
      - './ldap-persistence/backup:/data/backup'
      - './ldap-persistence/certs:/container/service/slapd/assets/certs'
    environment:
      - LDAP_BACKUP_CONFIG_CRON_EXP='0 4 * * *' # 每天早上4点运行备份slapd.d
      - LDAP_BACKUP_DATA_CRON_EXP='0 4 * * *' # 每天早上4点备份ldap
      - LDAP_BACKUP_TTL=15
      - LDAP_TLS_CRT_FILENAME=slapdcert.pem
      - LDAP_TLS_KEY_FILENAME=slapdkey.pem
      - LDAP_DOMAIN=yourdomain.org
      - LDAP_TLS=true
      - LDAP_ADMIN_PASSWORD=secret123
      - LDAP_CONFIG_PASSWORD=secret123
      #- KEEP_EXISTING_CONFIG=true
    hostname: ldap-service
    domainname: yourdomain.org
    #ports:
      #- 389:389 # 如果不想在这个docker-compose中架设起来phpldapadmin的话，就把这段解开，把下面给注释掉，然后使用apacheds来管理
  phpldapadmin-service:
    image: 'osixia/phpldapadmin:0.7.2'
    depends_on:
      - ldap-service
    volumes:
      - './ldap-persistence/certs:/container/service/slapd/assets/certs'
    hostname: phpldapadmin-service
    domainname: yourdomain.org
    hostname: phpldapadmin
    links:
      - ldap-service:ldap-host
    environment:
      - LDAP_TLS_CRT_FILENAME=slapdcert.pem
      - LDAP_TLS_KEY_FILENAME=slapdkey.pem
      - PHPLDAPADMIN_LDAP_CLIENT_TLS=true
      - PHPLDAPADMIN_LDAP_HOSTS=ldap-host
      - PHPLDAPADMIN_HTTPS=false
    ports:
      - '8000:80'  # 会在服务器上启动8000端口监听
```

顺便说下，389是openldap那个不加密的默认端口，所以只要在apacheds的连接上把apacheds的默认端口换成这个即可连接。默认的登录账户是 `cn=admin,dc=youdomain,dc=org`，默认密码是`admine`以及`config`。

就为了这个破docker-compose.yml 我弄了好长时间，还不知道那个加密的配置对不对，估计不太对。

## 管理，修改账户

[^using_apached_directory_studio_with_openldapd]: https://www.linux.com/learn/intro-to-linux/2017/2/how-install-apache-directory-studio-and-connect-openldap-server 
